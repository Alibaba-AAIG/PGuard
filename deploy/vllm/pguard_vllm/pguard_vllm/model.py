# Custom vLLM model architecture: Qwen3.5 backbone + PGuard safety heads.
#
# Registered under a distinct architecture name so the *original* backbone
# checkpoint loads unchanged; the guard heads come separately from
# PGUARD_HEADS / PGUARD_HEAD_PATH. Hidden states of the configured backbone
# layers are captured per step and scored per request using
# SchedulerOutput.num_scheduled_tokens for batch segmentation, which keeps the
# integration fully concurrent (no request serialization, no global locks
# around generation).
#
# The safety heads live in the heads/ package (GuardHeadRegistry, vLLM
# ModelRegistry style): this class only captures features, segments the batch
# per request, dispatches rows to each head's score_step and persists the
# returned vectors into the GuardStore. Multiple heads share the same
# captured hidden states (union of their capture layers) and are scored in
# parallel per step.
#
# Scoring is invoked from the patched GPUModelRunner.execute_model (see
# patch_runner.py) AFTER the engine step, for real scheduler steps only --
# this keeps the guard out of CUDA-graph capture and off the dummy-run path.
#
# Capture modes (PGUARD_CAPTURE_MODE):
#   hooks: forward hooks on the capture layers; eager mode only.
#   aux:   vLLM-native EAGLE3-style capture -- the model returns
#          (hidden_states, aux_hidden_states) and the runner's
#          use_aux_hidden_state_outputs machinery maintains fresh buffers
#          across CUDA-graph replays. Required for PGUARD_ENFORCE_EAGER=0.
#
# Per-request recurrent state lives inside each head's slot-pooled GPU
# buffers (see heads/base.py + state_pool.py): fixed memory, one
# index_select/index_put_ per step, slots freed (and zeroed) as soon as the
# scheduler reports the request finished.
#
# EngineCore runs in its own process; per-request score series are handed to
# the API process through the file-backed GuardStore (see store.py).
#
# Hot-path notes (see PGuard-Triton优化规划.md):
#   - capture layers resolve like the official deploy package:
#     PGUARD_CAPTURE_LAYERS env > config pguard_capture_layers >
#     selected_layers > feature_layers (the old order mis-fed the Lasttok
#     head early layers 0,1,2 instead of its trained 20,24,28);
#   - per-step feature gathering is one stack + one fp32 cast for the whole
#     batch (was per-request);
#   - scores reach the store as ONE host transfer per head per step (was one
#     .item() sync per scalar), and score files stay open per request (one
#     write syscall per step, no open/close churn);
#   - EMA prefill folding is closed-form (was a per-token Python loop);
#   - decode scoring can be thinned with PGUARD_COPY_LOGITS_PROB /
#     PGUARD_SCORE_INTERVAL (official cadence; default = score every step);
#     the EMA/GRU state update runs on EVERY decode step regardless;
#   - PGUARD_PREFILL_RISK_MODE=last scores only each request's last prefill
#     token (official default semantics; `all` keeps the full per-token
#     series for return_series queries).

import logging
import os

import torch

from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration

from .heads import build_head_from_checkpoint, resolve_head_dtype
from .pguard_config import get_config
from .store import get_store, resolve_head_dirs

logger = logging.getLogger("pguard_vllm")

_MAX_TRACKED_STATES = 4096
_EVICT_BACKOFF = 32  # exists() backstop sweep cadence (steps)

# counted degradation paths (see _bail): reason -> consecutive/total count
_BAILS: dict[str, int] = {}


def _bail(reason: str) -> None:
    """Record a guard step that produced NO scores.

    Degradation must leave traces (a silently disabled guard is a security
    failure): logged loudly on the first occurrence and every 1000th repeat;
    PGUARD_TRACE_GUARD=1 still gives per-step detail via _dbg."""
    n = _BAILS.get(reason, 0) + 1
    _BAILS[reason] = n
    if n == 1 or n % 1000 == 0:
        logger.error("pguard_vllm: guard step produced no scores (%s) x%d",
                     reason, n)

# raw PGuard model instance, resolved by patch_runner after engine load
_MODEL = None


def _parse_layers(spec, num_hidden_layers: int) -> list[int]:
    """Layer spec -> HF hidden_states indices ('all' -> 1..N-1)."""
    if isinstance(spec, str) and spec.strip().lower() == "all":
        return list(range(1, num_hidden_layers))
    if isinstance(spec, str):
        return [int(x) for x in spec.split(",") if x.strip()]
    return [int(x) for x in spec]


class Qwen3_5PGuardForConditionalGeneration(Qwen3_5ForConditionalGeneration):

    def __init__(self, *, vllm_config, prefix="model"):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.vllm_config = vllm_config
        self._pguard_ready = False
        self._capture_module_idx: list[int] = []
        self._cap: dict[int, torch.Tensor] = {}
        # hooks-mode stale-capture detector: under CUDA-graph replay the
        # Python forward never runs, so the captured tensors keep the SAME
        # object identity across steps (eager always produces fresh ones)
        self._cap_sig: tuple | None = None
        self._stale_steps = 0
        self.pguard_heads: dict[str, torch.nn.Module] = {}
        self.pguard_cfgs: dict[str, object] = {}
        # per-head read positions in the union capture stack (engine-side
        # bookkeeping; heads slice their own features at score time)
        self._layer_pos_by_head: dict[str, list[int]] = {}
        # requests whose prefill has been scored (decode-stage detection
        # without file IO)
        self._prefill_done: set[str] = set()
        # decode scoring cadence (official PGUARD_COPY_LOGITS_PROB semantics):
        # 1 = score every decode step; N = every Nth; <=0 = never (state still
        # updates). Prefill always scores.
        self._score_interval = 1
        self._decode_step = 0
        self._step_count = 0
        self._capture_mode = "hooks"
        # live status for the streaming interrupt check (server side):
        # {req_id: {head: {"plast": [...], "dmax": [...]}}}, rewritten to the
        # store's small live file once per step for the requests it changed
        self._live: dict[str, dict] = {}
        self._live_dirty: set[str] = set()

    # -- setup -------------------------------------------------------------
    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        try:
            self._setup_pguard()
        except Exception as e:
            logger.error("pguard_vllm: head setup failed (%s); "
                         "generation continues without guard scores\n%s",
                         e, __import__("traceback").format_exc())
        return loaded

    def _setup_pguard(self):
        global _MODEL
        if self._pguard_ready:
            return
        cfg = get_config()
        head_dirs = resolve_head_dirs()
        if not head_dirs:
            raise RuntimeError("no guard heads configured "
                               "(set PGUARD_HEADS or PGUARD_HEAD_PATH)")

        device = next(self.language_model.model.parameters()).device
        head_dtype = resolve_head_dtype(cfg.head_dtype)
        hf_indices = None
        num_layers = None
        per_head_indices = {}
        self.pguard_heads = {}
        self.pguard_cfgs = {}
        for name, path in head_dirs.items():
            head_cfg, head = build_head_from_checkpoint(path)
            self.pguard_heads[name] = head
            self.pguard_cfgs[name] = head_cfg
            num_layers = getattr(head_cfg, "num_hidden_layers", None) or 32
            # official resolution order (pguard/config.py +
            # guard/guard_utils.py): env override, then the checkpoint's
            # pguard_capture_layers. feature_layers is a TRAINING-side arg
            # and must only be a last-resort fallback.
            spec = (cfg.capture_layers
                    or getattr(head_cfg, "pguard_capture_layers", None)
                    or getattr(head_cfg, "selected_layers", None)
                    or getattr(head_cfg, "feature_layers", None)
                    or "all")
            indices = _parse_layers(spec, num_layers)
            per_head_indices[name] = indices
            hf_indices = sorted(set(hf_indices or []) | set(indices))

        self._capture_mode = cfg.capture_mode

        # Engine-side backstop for the pure `vllm serve` path (no server.py
        # preflight runs there): hooks capture needs eager -- graph replay
        # never executes the Python hooks and would rescore stale features.
        if self._capture_mode == "hooks" and not getattr(
                getattr(vllm_config, "model_config", None),
                "enforce_eager", True):
            raise RuntimeError(
                "pguard_vllm: capture_mode=hooks requires enforce_eager "
                "(CUDA-graph replay never runs Python hooks); set "
                "PGUARD_ENFORCE_EAGER=1 or PGUARD_CAPTURE_MODE=aux")

        if self._capture_mode == "aux":
            # vLLM-native capture (EAGLE3 machinery): the inner model appends
            # hidden_states + residual for these layers and returns
            # (hidden_states, aux); the runner keeps the buffers fresh across
            # CUDA-graph replays. patch_runner sets
            # use_aux_hidden_state_outputs=True accordingly.
            self.language_model.set_aux_hidden_state_layers(tuple(hf_indices))
        else:
            # one hook per union layer; each head reads its own subset
            layers = self.language_model.model.layers
            self._capture_module_idx = [i - 1 for i in hf_indices]
            for mi in self._capture_module_idx:
                layers[mi].register_forward_hook(self._make_hook(mi),
                                                 with_kwargs=True)
        self._layer_pos_by_head = {
            name: [hf_indices.index(i) for i in indices]
            for name, indices in per_head_indices.items()
        }

        # heads own their recurrent state + GPU-side aggregation; the engine
        # hands each head its flat-score layout and dispatches rows per step
        global guard_store
        guard_store = get_store(head_dirs)
        for name, head in self.pguard_heads.items():
            head.bind(device=device, dtype=head_dtype,
                      layer_pos=self._layer_pos_by_head[name],
                      total_cls=guard_store.total_cls(name),
                      task_layout=guard_store.task_layout(name)[0],
                      max_slots=cfg.max_slots)

        self._score_interval = cfg.score_interval
        self._decode_step = 0
        self._step_count = 0

        self._pguard_ready = True
        _MODEL = self
        logger.info("pguard_vllm: guard ready (heads=%s, capture hf-layers=%s, "
                    "mode=%s, dtype=%s, score_interval=%s, slots=%d, "
                    "device=%s)",
                    sorted(self.pguard_heads), hf_indices, self._capture_mode,
                    head_dtype, self._score_interval, cfg.max_slots, device)

    def _make_hook(self, module_idx: int):
        def hook(module, args, kwargs, output):
            if isinstance(output, tuple):
                h = output[0]
                r = output[1] if len(output) > 1 else None
            else:
                h, r = output, None
            if r is not None:
                h = h + r  # HF hidden_states[i] == residual stream after layer
            self._cap[module_idx] = h
        return hook

    # -- forward -----------------------------------------------------------
    def forward(self, input_ids, positions, intermediate_tensors=None,
                inputs_embeds=None, **kwargs):
        self._cap.clear()
        # In aux mode super().forward() returns (hidden_states, aux); the
        # _model_forward patch in patch_runner stashes the aux list for
        # run_guard_step. Scoring itself runs from the execute_model patch
        # after the engine step (never during graph capture / dummy runs).
        return super().forward(input_ids, positions, intermediate_tensors,
                               inputs_embeds, **kwargs)

    # -- scoring dispatch (called from patch_runner after the engine step) --
    def run_guard_step(self, sched: dict, flags: list | None, aux) -> None:
        def _dbg(msg):
            if get_config().trace_guard:
                try:
                    fd = os.open("/dev/shm/pguard_store/_guard_trace.log",
                                 os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
                    os.write(fd, ("%s\n" % msg).encode())
                    os.close(fd)
                except Exception:
                    pass

        _dbg("entry sched=%s cap=%s mode=%s" %
             (list(sched)[:2], sorted(self._cap), self._capture_mode))
        if not self._pguard_ready or not sched:
            _dbg("bail: not ready or empty sched")
            return
        total = sum(sched.values())
        if total == 0:
            _dbg("bail: total 0")
            return
        self._step_count += 1

        if self._capture_mode == "aux":
            caps = list(aux) if aux else None
            if not caps:
                _bail("aux_hidden_states_missing")
                return
        else:
            caps = [self._cap.get(mi) for mi in self._capture_module_idx]
            if not caps or any(c is None for c in caps):
                _bail("captured_layers_missing")
                return
            # stale-capture detection: identical tensor objects two steps in
            # a row mean the Python forward did not run (CUDA-graph replay)
            sig = tuple(id(c) for c in caps)
            if sig == self._cap_sig:
                self._stale_steps += 1
                if self._stale_steps >= 2:
                    _bail("stale_capture_graph_replay")
                    return
            else:
                self._cap_sig = sig
                self._stale_steps = 0
        feats_all = torch.stack(caps, dim=0).to(torch.float32)  # (C, T, H)
        if feats_all.shape[1] < total:
            _bail("captured_rows_short")
            return
        _dbg("scoring rows=%d total=%d" % (feats_all.shape[1], total))
        feats_all = feats_all[:, :total, :]

        if flags is None or len(flags) != len(sched):
            flags = self._decode_flags(list(sched.keys()), list(sched.values()))
        prefill_rows, decode_rows = [], []
        start = 0
        for (req_id, n), is_dec in zip(sched.items(), flags):
            sl = slice(start, start + n)
            feats = feats_all[:, sl, :]  # (C_union, n, H)
            if is_dec:
                decode_rows.append((req_id, feats))
            else:
                prefill_rows.append((req_id, feats))
                self._prefill_done.add(req_id)
            start += n

        # official-style cadence: counted per decode-bearing step; prefill
        # rows always score
        score = True
        if decode_rows and self._score_interval != 1:
            if self._score_interval <= 0:
                score = False
            else:
                self._decode_step += 1
                score = self._decode_step % self._score_interval == 0

        # one dispatch per head: heads return flat per-request vectors (one
        # D2H each), the engine persists them store-side and folds live status
        for name, head in self.pguard_heads.items():
            result = head.score_step(prefill_rows, decode_rows, score=score)
            if result is None:
                continue
            for rid, flat, n, is_dec in zip(result.req_ids, result.flats,
                                            result.ns, result.is_decodes):
                self._append_scores(name, rid, flat, n, is_dec)
        for rid in self._live_dirty:
            guard_store.write_live(rid, {h: {"plast": v["plast"], "dmax": v["dmax"]}
                                         for h, v in self._live[rid].items()})
        self._live_dirty.clear()
        self._evict_finished_states()

    def note_finished(self, req_ids) -> None:
        """Drop per-request guard state for requests the scheduler finished;
        write each request's final aggregated score record first (R2) so
        pop() is O(1)."""
        if not req_ids:
            return
        for rid in req_ids:
            self._prefill_done.discard(rid)
            self._live.pop(rid, None)
            self._live_dirty.discard(rid)
            guard_store.close_live(rid)
        for name, head in self.pguard_heads.items():
            for rid in req_ids:
                fin = head.finalize(rid)
                if fin is not None and (fin[0] or fin[1]):
                    guard_store.append_final(name, rid, fin[2], fin[3],
                                             fin[4], fin[0], fin[1])
                guard_store.close_fd(name, rid)

    # -- shared helpers ----------------------------------------------------
    def _append_scores(self, head: str, req_id: str, flat, n: int,
                       is_decode: bool):
        # flat: task-major class-probability vector(s) for this step's n
        # tokens (CPU list, length n * total_cls; see heads/base + store)
        guard_store.append(head, req_id, flat, n, is_decode)
        self._live_fold(head, req_id, flat, n, is_decode)

    @staticmethod
    def _top_risk(layout, blocks) -> float:
        """Highest non-normal class probability over per-task class slices
        (class 0 of every task is normal)."""
        best = 0.0
        for block in blocks:
            if len(block) > 1:
                best = max(best, max(block[1:]))
        return best

    def _live_fold(self, head: str, req_id: str, flat, n: int,
                   is_decode: bool) -> None:
        """Fold a scored step into the request's live status -- the same two
        vectors as its final record (plast = last prompt token, dmax = running
        max over scored output tokens), kept on the CPU copy the step already
        produced. The store file is rewritten only when the request's top
        non-normal probability rises: that is the only event that can turn a
        streaming check into a hit, at any threshold."""
        per_head = self._live.setdefault(req_id, {})
        vec = per_head.get(head)
        layout, _ = guard_store.task_layout(head)
        if not is_decode:
            # task i's block sits at [n*start, n*stop); its last token is
            # the block's tail
            plast = [v for task, start, stop in layout
                     for v in flat[n * start + (n - 1) * (stop - start):
                                          n * stop]]
            top = self._top_risk(layout, [plast[s:e] for _, s, e in layout])
            per_head[head] = {"plast": plast, "dmax": None,
                              "top": top, "written": top}
            self._live_dirty.add(req_id)
            return
        if vec is None:
            vec = per_head[head] = {"plast": None, "dmax": None,
                                    "top": -1.0, "written": -1.0}
        top = vec["top"]
        for t in range(n):
            row = [flat[n * start + t * (stop - start) + k]
                   for task, start, stop in layout
                   for k in range(stop - start)]
            cur = vec["dmax"]
            vec["dmax"] = row if cur is None else [
                a if a >= b else b for a, b in zip(cur, row)]
            top = max(top, self._top_risk(
                layout, [row[s:e] for _, s, e in layout]))
        vec["top"] = top
        if top > vec["written"]:      # a new high can only raise the verdict
            vec["written"] = top
            self._live_dirty.add(req_id)

    def _evict_finished_states(self):
        """Backstop cleanup: the scheduler's finished_req_ids handle the
        common case (note_finished); this file-existence sweep catches any
        other path at a low cadence. Bounded pools as safety valve."""
        if self._step_count % _EVICT_BACKOFF == 0:
            for name, head in self.pguard_heads.items():
                for req_id in head.tracked_req_ids():
                    if not any(guard_store.exists(h, req_id)
                               for h in self.pguard_heads):
                        head.free_req(req_id)
                        for h in self.pguard_heads:
                            guard_store.close_fd(h, req_id)
        for head in self.pguard_heads.values():
            head.evict_oldest_beyond(_MAX_TRACKED_STATES)

    def _decode_flags(self, req_ids: list[str], counts: list[int]) -> list[bool]:
        """Stage per batch row: True = decode step.

        Fallback only (the wrapper normally derives flags from the
        SchedulerOutput): an existing recurrent state or a scored prefill
        marks the request as past prefill.
        """
        flags = []
        for rid, n in zip(req_ids, counts):
            past = rid in self._prefill_done \
                or any(head.contains(rid)
                       for head in self.pguard_heads.values())
            flags.append(bool(past) and n == 1)
        return flags


# populated by Qwen3_5PGuardForConditionalGeneration._setup_pguard
# (kept annotation-free: this module is imported inside vLLM's architecture
#  inspection subprocess, where every name must already be defined)
guard_store = None


if __name__ == "__main__":
    # self-test: cadence dispatch through the head registry interface. A
    # prompt that joins while other requests are decoding keeps its Lasttok
    # prompt-side score on every step; only decode rows thin out, and EMA/GRU
    # state updates never thin out (the engine passes score=False; the heads
    # keep updating state).
    import types

    class _FakeHead:
        """Records the dispatch contract: (name, prefill rids, decode rids,
        score flag)."""

        def __init__(self, name, calls):
            self.name, self._calls = name, calls

        def score_step(self, prefill_rows, decode_rows, score=True):
            self._calls.append((self.name, [r for r, _ in prefill_rows],
                                [r for r, _ in decode_rows], bool(score)))
            return None

        def contains(self, rid):
            return False

    def _dispatch(interval, arrive):
        m = object.__new__(Qwen3_5PGuardForConditionalGeneration)
        m._pguard_ready, m._capture_mode, m._cap = True, "aux", {}
        m._capture_module_idx = [0, 1, 2]
        m._score_interval, m._decode_step, m._step_count = interval, 0, 0
        m._prefill_done = set()
        m._live, m._live_dirty = {}, set()
        calls = []
        m.pguard_heads = {n: _FakeHead(n, calls)
                          for n in ("lt", "ema", "gru")}
        m._evict_finished_states = lambda *a, **k: None
        one = [torch.zeros(1, 8)] * 3
        for _ in range(arrive - 1):     # advance the decode-step counter
            m.run_guard_step({"A": 1}, [True], one)
        calls.clear()
        # B's 5-token prompt is scheduled in the same step as A's decode token
        sched, flags = {"A": 1, "B": 5}, [True, False]
        aux = [torch.zeros(6, 8)] * 3
        m.run_guard_step(sched, flags, aux)
        return calls

    bad = []
    for arrive in (1, 4, 10):           # production default: nothing thins out
        calls = _dispatch(1, arrive)
        want = [("lt", ["B"], ["A"], True),
                ("ema", ["B"], ["A"], True),
                ("gru", ["B"], ["A"], True)]
        if calls != want:
            bad.append("interval=1 arrive=%d -> %r" % (arrive, calls))
    for arrive in range(1, 11):         # cadence 10: only step 10 scores decode
        calls = _dispatch(10, arrive)
        hit = arrive % 10 == 0
        want = [("lt", ["B"], ["A"], hit),
                ("ema", ["B"], ["A"], hit),
                ("gru", ["B"], ["A"], hit)]
        if calls != want:
            bad.append("interval=10 arrive=%d -> %r" % (arrive, calls))
    for b in bad:
        print("FAIL", b)

    # live status fold: plast = last prompt token, dmax = running max over
    # scored output tokens, both in global class order like the final record;
    # the store write happens only when the top non-normal probability rises
    globals()["guard_store"] = types.SimpleNamespace(
        task_layout=lambda head: ([("t_a", 0, 2), ("t_b", 2, 5)], 5),
        append=lambda *a: None, close_live=lambda *a: None)
    m = object.__new__(Qwen3_5PGuardForConditionalGeneration)
    m._live, m._live_dirty = {}, set()
    marks = []                      # store write needed after each step?
    for flat, n, is_dec in [        # top non-normal: 0.3 | 0.3 | 0.4 | 0.3
            ([0.9, 0.1, 0.8, 0.2, 0.7, 0.1, 0.2, 0.6, 0.3, 0.1], 2, False),
            ([0.0, 0.3, 0.5, 0.2, 0.3], 1, True),
            ([0.1, 0.2, 0.4, 0.4, 0.2], 1, True),
            ([0.0, 0.1, 0.4, 0.3, 0.2], 1, True)]:
        m._live_dirty.clear()
        m._append_scores("ema", "r", flat, n, is_dec)
        marks.append("r" in m._live_dirty)
    st = m._live["r"]["ema"]
    if (st["plast"], st["dmax"], st["top"], marks) != (
            [0.8, 0.2, 0.6, 0.3, 0.1], [0.1, 0.3, 0.5, 0.4, 0.3], 0.4,
            [True, False, True, False]):     # written only when the top rises
        bad.append("live status fold")
        print("FAIL live status fold:", st, "writes:", marks)
    print("model self-test (cadence dispatch, live status):",
          "FAILED (%d)" % len(bad) if bad else "ok")
    raise SystemExit(1 if bad else 0)
