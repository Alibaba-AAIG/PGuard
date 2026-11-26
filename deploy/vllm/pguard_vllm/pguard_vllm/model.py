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
# Multiple heads can run simultaneously: they share the same captured hidden
# states (same capture layers) and are scored in parallel per step. Available
# variants: lasttok (stateless), ema (per-request multi-scale EMA state),
# gru (query summary h0 + CfC response recurrence).
#
# Scoring is invoked from the patched GPUModelRunner.execute_model (see
# patch_runner.py) AFTER the engine step, for real scheduler steps only --
# this keeps the guard out of CUDA-graph capture and off the dummy-run path.
#
# Capture modes (PGUARD_CAPTURE_MODE):
#   hooks (default): forward hooks on the capture layers; eager mode only.
#   aux: vLLM-native EAGLE3-style capture -- the model returns
#        (hidden_states, aux_hidden_states) and the runner's
#        use_aux_hidden_state_outputs machinery maintains fresh buffers across
#        CUDA-graph replays. Required for PGUARD_ENFORCE_EAGER=0.
#
# Per-request recurrent state lives in slot-pooled GPU buffers (see
# state_pool.py): fixed memory, one index_select/index_put_ per step, slots
# freed (and zeroed) as soon as the scheduler reports the request finished.
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

from .head import build_head_from_checkpoint
from .state_pool import StatePool
from .store import get_store, resolve_head_dirs

logger = logging.getLogger("pguard_vllm")

_MAX_TRACKED_STATES = 4096
_EVICT_BACKOFF = 32  # exists() backstop sweep cadence (steps)

_HEAD_DTYPES = {
    "float32": torch.float32, "fp32": torch.float32, "": torch.float32,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "float16": torch.float16, "fp16": torch.float16,
}

# raw PGuard model instance, resolved by patch_runner after engine load
_MODEL = None


def _parse_layers(spec, num_hidden_layers: int) -> list[int]:
    """Layer spec -> HF hidden_states indices ('all' -> 1..N-1)."""
    if isinstance(spec, str) and spec.strip().lower() == "all":
        return list(range(1, num_hidden_layers))
    if isinstance(spec, str):
        return [int(x) for x in spec.split(",") if x.strip()]
    return [int(x) for x in spec]


def _ema_fold(alphas, feats, state):
    """Exact multi-scale causal EMA fold, closed form:

        ema_n = sum_t a*(1-a)^(n-1-t)*x_t + (1-a)^n * ema_0

    alphas: (n_scales, H) fp32; feats: (C, n, H) fp32; state: (C, n_scales, H)
    fp32. Replaces the per-token Python loop (n kernel-launch rounds) with one
    weighted-sum einsum; same math as the official torch fallback
    (pguard/guard/history_state.py prefill_update).
    """
    C, n, H = feats.shape
    if n == 0:
        return state
    ret = 1.0 - alphas  # (n_scales, H)
    if n == 1:
        return (alphas.unsqueeze(0) * feats[:, 0, :].unsqueeze(1)
                + ret.unsqueeze(0) * state)
    # weight of token j (0-based): a * (1-a)^(n-1-j)
    k = torch.arange(n - 1, -1, -1, device=feats.device, dtype=torch.float32)
    w = alphas.unsqueeze(-1) * torch.exp(torch.log(ret).unsqueeze(-1) * k)
    within = torch.einsum("shn,cnh->csh", w, feats)  # (C, n_scales, H)
    decay = torch.exp(torch.log(ret) * float(n))     # (n_scales, H)
    return within + decay.unsqueeze(0) * state


class Qwen3_5PGuardForConditionalGeneration(Qwen3_5ForConditionalGeneration):

    def __init__(self, *, vllm_config, prefix="model"):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._pguard_ready = False
        self._capture_module_idx: list[int] = []
        self._cap: dict[int, torch.Tensor] = {}
        self.pguard_heads: dict[str, torch.nn.Module] = {}
        self.pguard_cfgs: dict[str, object] = {}
        # slot-pooled per-request recurrent state, one pool per head:
        #   ema: (C, n_scales, hidden) running EMA vectors per slot
        #   gru: (mem_dim,) response recurrence hidden state per slot
        self._pools: dict[str, StatePool] = {}
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
        # prefill scoring breadth for token-wise heads (lasttok):
        # all = every prefill token (enables prompt_series); last = each
        # request's final token only (official PGUARD_PREFILL_RISK_MODE=last)
        self._prefill_risk_mode = "all"
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
        head_dirs = resolve_head_dirs()
        if not head_dirs:
            raise RuntimeError("no guard heads configured "
                               "(set PGUARD_HEADS or PGUARD_HEAD_PATH)")

        device = next(self.language_model.model.parameters()).device
        head_dtype = _HEAD_DTYPES.get(
            os.environ.get("PGUARD_HEAD_DTYPE", "").strip().lower(),
            torch.float32)
        hf_indices = None
        num_layers = None
        per_head_indices = {}
        for name, path in head_dirs.items():
            cfg, head = build_head_from_checkpoint(path)
            head = head.to(device=device, dtype=head_dtype)
            self.pguard_heads[name] = head
            self.pguard_cfgs[name] = cfg
            num_layers = getattr(cfg, "num_hidden_layers", None) or 32
            # official resolution order (pguard/config.py +
            # guard/guard_utils.py): env override, then the checkpoint's
            # pguard_capture_layers. feature_layers is a TRAINING-side arg
            # and must only be a last-resort fallback.
            spec = (os.environ.get("PGUARD_CAPTURE_LAYERS", "").strip()
                    or getattr(cfg, "pguard_capture_layers", None)
                    or getattr(cfg, "selected_layers", None)
                    or getattr(cfg, "feature_layers", None)
                    or "all")
            indices = _parse_layers(spec, num_layers)
            per_head_indices[name] = indices
            hf_indices = sorted(set(hf_indices or []) | set(indices))

        self._capture_mode = (os.environ.get("PGUARD_CAPTURE_MODE", "hooks")
                              .strip().lower())
        if self._capture_mode not in ("hooks", "aux"):
            self._capture_mode = "hooks"

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

        # slot-pooled recurrent state + score accumulators, one pool per head
        global guard_store
        guard_store = get_store(head_dirs)
        max_slots = int(os.environ.get("PGUARD_MAX_SLOTS", "64"))
        self._pools = {}
        for name, head in self.pguard_heads.items():
            kind = self.pguard_cfgs[name].pguard_version
            if kind == "ema":
                n_scales = head.ema_fusion.ema_module.alphas.shape[0]
                ema_shape = (len(self._layer_pos_by_head[name]), n_scales,
                             next(head.parameters()).shape[-1])
                self._pools[name] = StatePool(name, device,
                                              ema_shape=ema_shape,
                                              max_slots=max_slots,
                                              total_cls=guard_store.total_cls(name))
            elif kind == "gru":
                self._pools[name] = StatePool(name, device,
                                              mem_dim=head.mem_dim,
                                              max_slots=max_slots,
                                              total_cls=guard_store.total_cls(name))
            else:
                # stateless head: no recurrent state, accumulator only
                self._pools[name] = StatePool(name, device,
                                              max_slots=max_slots,
                                              total_cls=guard_store.total_cls(name))

        self._prefill_risk_mode = (os.environ.get("PGUARD_PREFILL_RISK_MODE",
                                                  "all").strip().lower())
        if self._prefill_risk_mode not in ("all", "last"):
            self._prefill_risk_mode = "all"

        prob = os.environ.get("PGUARD_COPY_LOGITS_PROB", "1.0").strip()
        interval_env = os.environ.get("PGUARD_SCORE_INTERVAL", "").strip()
        if interval_env:
            self._score_interval = max(0, int(interval_env))
        else:
            try:
                prob = float(prob)
            except ValueError:
                prob = 1.0
            self._score_interval = (0 if prob <= 0.0 else
                                    1 if prob >= 1.0 else
                                    max(1, round(1.0 / prob)))
        self._decode_step = 0
        self._step_count = 0

        self._pguard_ready = True
        _MODEL = self
        logger.info("pguard_vllm: guard ready (heads=%s, capture hf-layers=%s, "
                    "mode=%s, dtype=%s, score_interval=%s, prefill_mode=%s, "
                    "slots=%d, device=%s)",
                    sorted(self.pguard_heads), hf_indices, self._capture_mode,
                    head_dtype, self._score_interval, self._prefill_risk_mode,
                    max_slots, device)

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
            if os.environ.get("PGUARD_TRACE_GUARD", "") == "1":
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
                _dbg("bail: no aux")
                return
        else:
            caps = [self._cap.get(mi) for mi in self._capture_module_idx]
            if not caps or any(c is None for c in caps):
                _dbg("bail: caps missing %s/%s" %
                     (len(self._cap), len(self._capture_module_idx)))
                return
        feats_all = torch.stack(caps, dim=0).to(torch.float32)  # (C, T, H)
        if feats_all.shape[1] < total:
            _dbg("bail: rows %d < total %d" % (feats_all.shape[1], total))
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

        for name, head in self.pguard_heads.items():
            kind = self.pguard_cfgs[name].pguard_version
            pos = self._layer_pos_by_head[name]
            if kind == "lasttok":
                # prefill rows always score; only decode rows follow the cadence
                lt_decode = decode_rows if score else []
                if prefill_rows or lt_decode:
                    self._score_lasttok(name, head, pos,
                                        prefill_rows + lt_decode,
                                        [False] * len(prefill_rows)
                                        + [True] * len(lt_decode))
            elif kind == "ema":
                if prefill_rows:
                    self._ema_prefill(name, head, pos, prefill_rows)
                if decode_rows:
                    self._ema_decode(name, head, pos, decode_rows, score=score)
            elif kind == "gru":
                if prefill_rows:
                    self._gru_prefill(name, head, pos, prefill_rows)
                if decode_rows:
                    self._gru_decode(name, head, pos, decode_rows, score=score)
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
        for head, pool in self._pools.items():
            for rid in req_ids:
                if not pool.contains(rid):
                    continue
                hp, hd, plast, dlast, dmax = pool.acc_read(pool.slot_of[rid])
                if hp or hd:
                    guard_store.append_final(head, rid, plast, dlast, dmax,
                                             hp, hd)
                guard_store.close_fd(head, rid)
        for pool in self._pools.values():
            for rid in req_ids:
                pool.free_req(rid)

    # -- shared helpers ----------------------------------------------------
    def _append_scores(self, head: str, req_id: str, flat, n: int,
                       is_decode: bool):
        # flat: task-major class-probability vector(s) for this step's n
        # tokens (CPU list, length n * total_cls; see state_pool/store)
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

    def _acc_update(self, head_name: str, req_ids: list[str], flats, ns,
                    is_decodes) -> None:
        """GPU-side running aggregation (R2): fold each request's newest
        scored-token vector into its slot accumulator (one small kernel chain
        per scoring step). Read back once on finish for the final record."""
        pool = self._pools.get(head_name)
        if pool is None or pool.acc_last_p is None or not flats:
            return
        layout, _ = guard_store.task_layout(head_name)
        device = flats[0].device
        slot_t = pool.slots_for(req_ids, device)
        last_vecs = []
        for f, n in zip(flats, ns):
            parts = []
            for task, start, stop in layout:
                n_cls = stop - start
                # task i's block sits at [n*start, n*stop); its last token is
                # the block's tail
                parts.append(f[n * start + (n - 1) * n_cls:n * stop])
            last_vecs.append(torch.cat(parts))
        pool.acc_update(slot_t, torch.stack(last_vecs), is_decodes)

    def _probs_to_cpu(self, head_name: str, probs_per_req, req_ids: list[str],
                      is_decodes):
        """Move a per-request stack of probability tensors to the host with
        ONE device-to-host transfer, as flat task-major vectors
        ([task0 tok0..tokN | task1 ...], length n * total_cls per request;
        layout mirrors the official interrupt_policy column ranges). Also
        folds the newest scored-token vectors into the GPU accumulators."""
        total = guard_store.total_cls(head_name)
        flats, ns = [], []
        for probs in probs_per_req:
            flats.append(torch.cat([p.reshape(-1) for p in probs]))
            ns.append(int(probs[0].shape[0]) if probs else 0)
        if not flats:
            return []
        self._acc_update(head_name, req_ids, flats, ns, is_decodes)
        flat = torch.cat(flats).tolist()  # single D2H transfer
        out, ofs = [], 0
        for n in ns:
            out.append(flat[ofs:ofs + n * total])
            ofs += n * total
        return out

    def _evict_finished_states(self):
        """Backstop cleanup: the scheduler's finished_req_ids handle the
        common case (note_finished); this file-existence sweep catches any
        other path at a low cadence. Bounded pools as safety valve."""
        if self._step_count % _EVICT_BACKOFF == 0:
            for pool in self._pools.values():
                for req_id in pool.req_ids():
                    if not any(guard_store.exists(head, req_id)
                               for head in self.pguard_heads):
                        pool.free_req(req_id)
                        for head in self.pguard_heads:
                            guard_store.close_fd(head, req_id)
        for pool in self._pools.values():
            while len(pool.slot_of) > _MAX_TRACKED_STATES:
                pool.free_req(pool.req_ids()[0])

    def _decode_flags(self, req_ids: list[str], counts: list[int]) -> list[bool]:
        """Stage per batch row: True = decode step.

        Fallback only (the wrapper normally derives flags from the
        SchedulerOutput): an existing recurrent state or a scored prefill
        marks the request as past prefill.
        """
        flags = []
        for rid, n in zip(req_ids, counts):
            past = rid in self._prefill_done \
                or any(pool.contains(rid) for pool in self._pools.values())
            flags.append(bool(past) and n == 1)
        return flags

    def _pad_rows(self, rows):
        """Pad ragged (C, n_i, H) row features into (B, C, max_n, H) + mask."""
        C = rows[0][1].shape[0]
        H = rows[0][1].shape[2]
        max_n = max(f.shape[1] for _, f in rows)
        device = rows[0][1].device
        padded = torch.zeros(len(rows), C, max_n, H, device=device,
                             dtype=torch.float32)
        mask = torch.zeros(len(rows), max_n, dtype=torch.bool, device=device)
        for i, (_, f) in enumerate(rows):
            padded[i, :, :f.shape[1], :] = f
            mask[i, :f.shape[1]] = True
        return padded, mask

    # -- lasttok (stateless) ----------------------------------------------
    def _score_lasttok(self, head_name: str, head, layer_pos, rows, decode_flags):
        rows = [(rid, f[layer_pos]) for rid, f in rows]
        if self._prefill_risk_mode == "last":
            # score only each request's final token (decode rows are n=1, so
            # this is a no-op for them)
            rows = [(rid, f[:, -1:, :]) for rid, f in rows]
        if all(f.shape[1] == 1 for _, f in rows):
            # every row already has exactly one token
            padded = torch.stack([f for _, f in rows], dim=0)  # (B, C, 1, H)
        else:
            padded, _mask = self._pad_rows(rows)
        hidden = padded.to(dtype=self._head_dtype(head_name))
        with torch.no_grad():
            out = head(features=hidden, token_type_ids=None,
                       label_type=None, label_marker=None)
        probs = [torch.softmax(l.float(), dim=-1)
                 for l in out.full_classification_logits]  # (B, n, n_cls)
        ns = [f.shape[1] for _, f in rows]
        req_ids = [rid for rid, _ in rows]
        cpu = self._probs_to_cpu(head_name, [[p[i, :n] for p in probs]
                                             for i, n in enumerate(ns)],
                                 req_ids, decode_flags)
        for i, (req_id, _) in enumerate(rows):
            self._append_scores(head_name, req_id, cpu[i], ns[i],
                                decode_flags[i])

    # -- ema (seq) ---------------------------------------------------------
    def _ema_alphas(self, head):
        """(n_scales, H) fp32 alphas on the head's device, cached per head."""
        cached = getattr(head, "_pguard_alphas", None)
        device = next(head.parameters()).device
        if cached is not None and cached.device == device:
            return cached
        alphas = head.ema_fusion.ema_module.alphas.to(device=device,
                                                      dtype=torch.float32)
        head._pguard_alphas = alphas
        return alphas

    def _head_dtype(self, head_name: str):
        return next(self.pguard_heads[head_name].parameters()).dtype

    def _ema_prefill(self, head_name: str, head, layer_pos, rows):
        """Fold each prefill chunk into the request's EMA state (closed-form
        causal recurrence, one weighted-sum kernel per chunk instead of a
        per-token Python loop); score the chunk's LAST token only (the API
        aggregates prompt-side risk from the last prefill record)."""
        alphas = self._ema_alphas(head)  # (n_scales, H) fp32
        pool = self._pools[head_name]
        slot_t = pool.slots_for([rid for rid, _ in rows],
                                rows[0][1].device)
        states_b = pool.ema_read(slot_t)  # (B, C, n_scales, H) fp32, zeros for new
        new_states, last_feats = [], []
        for i, (req_id, feats_all) in enumerate(rows):
            feats = feats_all[layer_pos]  # (C, n, H) fp32
            state = _ema_fold(alphas, feats, states_b[i])
            new_states.append(state)
            last_feats.append(feats[:, -1, :])
        states_b = torch.stack(new_states, dim=0)             # (B, C, n_scales, H)
        pool.ema_write(slot_t, states_b)
        feats_b = torch.stack(last_feats, dim=0).unsqueeze(2)  # (B, C, 1, H)
        logits = self._ema_score_batch(head_name, head, states_b, feats_b)
        probs_all = [torch.softmax(l.float()[:, 0], dim=-1) for l in logits]
        cpu = self._probs_to_cpu(head_name, [[p[i].unsqueeze(0) for p in probs_all]
                                             for i in range(len(rows))],
                                 [rid for rid, _ in rows],
                                 [False] * len(rows))
        for i, (req_id, _) in enumerate(rows):
            self._append_scores(head_name, req_id, cpu[i], 1, is_decode=False)

    def _ema_score_batch(self, head_name: str, head, batch_states, feats_b):
        """Gate -> merge -> fuse -> classify for a batch of tokens.

        batch_states: (B, C, n_scales, H); feats_b: (B, C, 1, H). Returns
        per-task logits of shape (B, 1, n_cls)."""
        dtype = self._head_dtype(head_name)
        feats = feats_b.to(dtype)
        B, C, _, H = feats.shape
        scores = head.gate(head.gate_norm(feats)).squeeze(-1)  # (B, C, 1)
        if head.top_L >= C:
            # all captured layers are kept: the top-k gather is a no-op, the
            # weighted sum over the softmax weights is identical
            w = torch.softmax(scores, dim=1).unsqueeze(-1)     # (B, C, 1, 1)
            merged_cur = (feats * w).sum(dim=1)                # (B, 1, H)
            st = batch_states.to(dtype)
            merged_ema = (st * w).sum(dim=1)                   # (B, n_scales, H)
        else:
            topk_scores, topk_indices = torch.topk(scores, k=head.top_L, dim=1)
            w = torch.softmax(topk_scores, dim=1)  # (B, topk, 1)
            merged_cur = (feats.gather(1, topk_indices.unsqueeze(-1)
                                       .expand(-1, -1, 1, H))
                          * w.unsqueeze(-1)).sum(dim=1)  # (B, 1, H)
            st = batch_states.to(dtype)
            st_sel = st.gather(1, topk_indices.unsqueeze(-1)
                               .expand(-1, -1, st.shape[2], st.shape[3]))
            merged_ema = (st_sel * w.unsqueeze(2)).sum(dim=1)  # (B, n_scales, H)
        ema_heads = [merged_ema[:, i].unsqueeze(1)
                     for i in range(merged_ema.shape[1])]
        fused = head.ema_fusion.fuse(ema_heads, merged_cur)  # (B, 1, H)
        h = head.norm(fused)
        h = head.input_proj(h)
        for layer in head.adapter:
            h = layer(h)
        return [classifier(h) for classifier in head.classifiers]

    def _ema_decode(self, head_name: str, head, layer_pos, rows, score=True):
        """Batched single-token decode: the EMA state update runs on EVERY
        step; scoring can be thinned via score=False (cadence)."""
        alphas = self._ema_alphas(head)
        pool = self._pools[head_name]
        device = rows[0][1].device
        slot_t = pool.slots_for([rid for rid, _ in rows], device)
        batch_states = pool.ema_read(slot_t)  # (B, C, n_scales, H) fp32

        feats_b = torch.stack([f[layer_pos] for _, f in rows], dim=0)  # (B, C, 1, H)
        xi = feats_b.squeeze(2)  # (B, C, H)
        batch_states = alphas.unsqueeze(0).unsqueeze(0) * xi.unsqueeze(2) \
            + (1.0 - alphas.unsqueeze(0).unsqueeze(0)) * batch_states

        pool.ema_write(slot_t, batch_states)
        if not score:
            return
        B = len(rows)
        logits = self._ema_score_batch(head_name, head, batch_states, feats_b)
        probs_all = [torch.softmax(l.float()[:, 0], dim=-1) for l in logits]
        cpu = self._probs_to_cpu(head_name, [[p[i].unsqueeze(0) for p in probs_all]
                                             for i in range(B)],
                                 [rid for rid, _ in rows], [True] * B)
        for i, (req_id, _) in enumerate(rows):
            self._append_scores(head_name, req_id, cpu[i], 1, is_decode=True)

    # -- gru (cfc) ---------------------------------------------------------
    def _gru_prefill(self, head_name: str, head, layer_pos, rows):
        """Batched query summarization -> h0 state + query-level scores."""
        rows = [(rid, f[layer_pos]) for rid, f in rows]
        padded, mask = self._pad_rows(rows)
        feats = padded.to(dtype=self._head_dtype(head_name))
        with torch.no_grad():
            query_logits_list, h0, _ = head.forward_prefill(feats, mask=mask)
        probs_all = [torch.softmax(ql.float(), dim=-1)
                     for ql in query_logits_list]  # (B, n_cls)
        cpu = self._probs_to_cpu(head_name, [[p[i].unsqueeze(0) for p in probs_all]
                                             for i in range(len(rows))],
                                 [rid for rid, _ in rows],
                                 [False] * len(rows))
        pool = self._pools[head_name]
        slot_t = pool.slots_for([rid for rid, _ in rows],
                                rows[0][1].device)
        pool.gru_write(slot_t, h0)
        for i, (req_id, _) in enumerate(rows):
            self._append_scores(head_name, req_id, cpu[i], 1, is_decode=False)

    def _gru_decode(self, head_name: str, head, layer_pos, rows, score=True):
        """Batched single-token CfC/GRU step with carried hidden state; the
        state recurrence runs every step, scoring can be thinned."""
        pool = self._pools[head_name]
        device = rows[0][1].device
        slot_t = pool.slots_for([rid for rid, _ in rows], device)
        h_prev = pool.gru_read(slot_t)  # (B, mem_dim) fp32
        feats = torch.stack([f[layer_pos] for _, f in rows],
                            dim=0).to(self._head_dtype(head_name))
        with torch.no_grad():
            logits_list, h_new = head.forward_step(feats, h_prev,
                                                   compute_scores=score)
        pool.gru_write(slot_t, h_new)
        if not score:
            return
        probs_all = [torch.softmax(l.float()[:, 0], dim=-1)
                     for l in logits_list]  # (B, n_cls)
        cpu = self._probs_to_cpu(head_name, [[p[i].unsqueeze(0) for p in probs_all]
                                             for i in range(len(rows))],
                                 [rid for rid, _ in rows], [True] * len(rows))
        for i, (req_id, _) in enumerate(rows):
            self._append_scores(head_name, req_id, cpu[i], 1, is_decode=True)


# populated by Qwen3_5PGuardForConditionalGeneration._setup_pguard
# (kept annotation-free: this module is imported inside vLLM's architecture
#  inspection subprocess, where every name must already be defined)
guard_store = None


if __name__ == "__main__":
    # self-test: cadence dispatch. A prompt that joins while other requests
    # are decoding keeps its Lasttok prompt-side score on every step; only
    # decode rows thin out, and EMA/GRU state updates never thin out.
    import types

    def _dispatch(interval, arrive):
        m = object.__new__(Qwen3_5PGuardForConditionalGeneration)
        m._pguard_ready, m._capture_mode, m._cap = True, "aux", {}
        m._capture_module_idx = [0, 1, 2]
        m._score_interval, m._decode_step, m._step_count = interval, 0, 0
        m._prefill_done, m._prefill_risk_mode = set(), "all"
        m._live, m._live_dirty = {}, set()
        m.pguard_heads = {"lt": object(), "ema": object(), "gru": object()}
        m.pguard_cfgs = {k: types.SimpleNamespace(pguard_version=v) for k, v in
                         (("lt", "lasttok"), ("ema", "ema"), ("gru", "gru"))}
        m._layer_pos_by_head = {"lt": 0, "ema": 1, "gru": 2}
        calls = []

        def _rec(name):
            return lambda *a, **k: calls.append(
                (name, [r[0] for r in a[3]], k.get("score", True)))

        for name in ("_score_lasttok", "_ema_prefill", "_ema_decode",
                     "_gru_prefill", "_gru_decode"):
            setattr(m, name, _rec(name))
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

    def _lt_prompt_scored(calls):
        return "B" in [rid for name, rids, _ in calls
                       if name == "_score_lasttok" for rid in rids]

    bad = []
    for arrive in (1, 4, 10):           # production default: nothing thins out
        calls = _dispatch(1, arrive)
        lt = [rids for name, rids, _ in calls if name == "_score_lasttok"]
        if not (_lt_prompt_scored(calls) and lt == [["B", "A"]]
                and all(s for _, _, s in calls)):
            bad.append("interval=1 arrive=%d" % arrive)
    for arrive in range(1, 11):         # cadence 10: only step 10 scores decode
        calls = _dispatch(10, arrive)
        hit = arrive % 10 == 0
        dec = [(n, rids, s) for n, rids, s in calls if n.endswith("_decode")]
        if not (_lt_prompt_scored(calls)
                and any(n == "_ema_prefill" and "B" in rids
                        for n, rids, _ in calls)
                and len(dec) == 2
                and all(rids == ["A"] and s == hit for _, rids, s in dec)):
            bad.append("interval=10 arrive=%d" % arrive)
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
