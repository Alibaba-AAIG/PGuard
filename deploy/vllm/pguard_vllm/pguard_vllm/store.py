# Per-request guard-score accumulation and aggregation.
#
# The safety heads run inside vLLM's EngineCore process, which is a SEPARATE
# process from the API server even with VLLM_ENABLE_V1_MULTIPROCESSING=0
# (AsyncLLM hardcodes the MP client in 0.25.1). Scores are therefore handed
# over through small append-only files under /dev/shm: the engine appends one
# JSON line per scored step, the API reads + aggregates + unlinks on pop().
# One file per (head, request); O_APPEND keeps lines atomic.
#
# Multiple heads can run simultaneously in the same engine: each head writes
# to its own file (<head>__<req_id>.jsonl) and the API aggregates all of them.

import glob
import json
import logging
import os
import shutil

logger = logging.getLogger("pguard_vllm")

STORE_DIR = os.environ.get("PGUARD_STORE_DIR", "/dev/shm/pguard_store")


def load_task_schema(head_path: str):
    """(classification_heads, id2label) from the head checkpoint config.json."""
    with open(os.path.join(head_path, "config.json")) as f:
        cfg = json.load(f)
    heads = cfg["classifier"]["classification_heads"]  # [[name, n_classes], ...]
    id2label = {name: schema["id2label"] for name, schema in cfg["task_schema"].items()}
    return heads, id2label


def head_name_from_dir(subdir_or_path: str) -> str:
    """'PGuard-EMA-Qwen3.5-9B' -> 'ema' (variant name used as the head key)."""
    base = os.path.basename(subdir_or_path.rstrip("/"))
    parts = base.split("-")
    return parts[1].lower() if len(parts) > 1 else base.lower()


def resolve_head_dirs() -> dict:
    """{head_name: checkpoint_dir} from PGUARD_HEADS / PGUARD_HEAD_SUBDIR /
    PGUARD_HEAD_PATH (first match wins per name)."""
    dirs = {}
    single = os.environ.get("PGUARD_HEAD_PATH")
    if single:
        dirs[head_name_from_dir(single)] = single
    sub = os.environ.get("PGUARD_HEAD_SUBDIR")
    if sub:
        dirs[head_name_from_dir(sub)] = os.path.join("/pguard_heads", sub)
    multi = os.environ.get("PGUARD_HEADS")
    if multi:
        for sub in multi.split(","):
            sub = sub.strip()
            if sub:
                dirs[head_name_from_dir(sub)] = os.path.join("/pguard_heads", sub)
    return dirs


def _req_path(store_dir: str, head: str, req_id: str) -> str:
    return os.path.join(store_dir, f"{head}__{req_id}.jsonl")


def _find_path(store_dir: str, head: str, req_id: str) -> str | None:
    """Exact file first; the engine may append a suffix to the request id
    (observed with vLLM 0.25.1's async request path), so fall back to prefix
    matching over the live files."""
    exact = _req_path(store_dir, head, req_id)
    if os.path.exists(exact):
        return exact
    matches = sorted(glob.glob(os.path.join(store_dir, f"{head}__{req_id}*.jsonl")))
    return matches[0] if matches else None


class GuardStore:
    """File-backed per-(head, request) score store shared across processes."""

    def __init__(self, head_dirs: dict, store_dir: str | None = None):
        self.head_dirs = dict(head_dirs)
        self.store_dir = store_dir or STORE_DIR
        os.makedirs(self.store_dir, exist_ok=True)
        self._schemas = {}
        self._layouts = {}
        # engine-side fd cache: one open per (head, request) for its whole
        # lifetime instead of open/close on every scored step
        self._fds = {}

    def schema(self, head: str):
        if head not in self._schemas:
            path = self.head_dirs.get(head)
            if not path:
                raise RuntimeError(f"no checkpoint dir configured for head {head!r}")
            self._schemas[head] = load_task_schema(path)
        return self._schemas[head]

    def task_layout(self, head: str):
        """Flat-score layout for a head: ([(task, start, stop)] class offsets
        into the concatenated 38-dim vector, total classes). Mirrors the
        official interrupt_policy HeadRules column ranges."""
        if head not in self._layouts:
            heads, _ = self.schema(head)
            layout, start = [], 0
            for task, n_cls in heads:
                layout.append((task, start, start + n_cls))
                start += n_cls
            self._layouts[head] = (layout, start)
        return self._layouts[head]

    def total_cls(self, head: str) -> int:
        return self.task_layout(head)[1]

    @property
    def heads(self):
        return list(self.head_dirs)

    # -- writer side (engine core process) ---------------------------------
    def append(self, head: str, req_id: str, flat, n: int, is_decode: bool) -> None:
        """Append one scored step as a flat task-major class vector.

        flat: concatenated per-task probabilities for the step's n tokens
        ([task0 tok0..tokN | task1 ...], length n * total_cls, CPU list);
        n: number of scored tokens in this step."""
        payload = {
            "s": "decode" if is_decode else "prefill",
            "n": int(n),
            "p": [round(v, 6) for v in flat],
        }
        line = json.dumps(payload, separators=(",", ":"))
        key = (head, req_id)
        fd = self._fds.get(key)
        if fd is None:
            fd = os.open(_req_path(self.store_dir, head, req_id),
                         os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
            self._fds[key] = fd
        os.write(fd, (line + "\n").encode())

    def append_final(self, head: str, req_id: str, plast, dlast, dmax,
                     has_p: bool, has_d: bool) -> None:
        """Write the request's final aggregated record (R2): pop() reads this
        single line (O(1)) instead of replaying every per-step record.
        plast/dlast/dmax are flat class vectors (or None when that stage was
        never scored)."""
        payload = {
            "s": "final",
            "plast": plast if has_p else None,
            "dlast": dlast if has_d else None,
            "dmax": dmax if has_d else None,
        }
        line = json.dumps(payload, separators=(",", ":"))
        key = (head, req_id)
        fd = self._fds.get(key)
        if fd is None:
            fd = os.open(_req_path(self.store_dir, head, req_id),
                         os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
            self._fds[key] = fd
        os.write(fd, (line + "\n").encode())

    def close_fd(self, head: str, req_id: str) -> None:
        """Close a cached fd (called when the request finishes). Writes to an
        fd whose file the API already unlinked are silently discarded."""
        fd = self._fds.pop((head, req_id), None)
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass

    def close_all_fds(self) -> None:
        for fd in self._fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        self._fds.clear()

    def exists(self, head: str, req_id: str) -> bool:
        """Cheap finished-check for the engine hot path: append() always
        creates the file at the exact path, and pop() unlinks it, so file
        presence alone tracks "request still scoring". Avoids the full
        read+JSON-parse of peek() on every decode step."""
        return os.path.exists(_req_path(self.store_dir, head, req_id))

    def has_prefill(self, head: str, req_id: str) -> bool:
        path = _find_path(self.store_dir, head, req_id)
        if path is None:
            return False
        try:
            with open(path) as f:
                return any('"s":"prefill"' in line or '"s": "prefill"' in line
                           for line in f)
        except FileNotFoundError:
            return False

    # -- reader side (api server process) ----------------------------------
    def _read(self, head: str, req_id: str):
        path = _find_path(self.store_dir, head, req_id)
        if path is None:
            return None
        try:
            with open(path) as f:
                return [json.loads(line) for line in f if line.strip()]
        except FileNotFoundError:
            return None

    def latest(self, head: str, req_id: str):
        """Last scored record {'s': stage, 'p': [[cls...], ...]} or None.
        Cheap per-chunk read for streaming interrupt checks."""
        path = _find_path(self.store_dir, head, req_id)
        if path is None:
            return None
        try:
            with open(path, 'rb') as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - 4096))
                lines = f.read().decode().strip().split('\n')
            for line in reversed(lines):
                if line.strip():
                    return json.loads(line)
        except (FileNotFoundError, json.JSONDecodeError):
            return None
        return None

    def flat_to_rows(self, head: str, n: int, flat):
        """Convert a step's flat task-major vector ([task0 tok0..tokN |
        task1 ...]) into n token-major rows (each total_cls long, in global
        class order). n = number of scored tokens in the step."""
        layout, total = self.task_layout(head)
        rows = [[] for _ in range(n)]
        for task, start, stop in layout:
            n_cls = stop - start
            blk = flat[n * start:n * stop]
            for t in range(n):
                rows[t].extend(blk[t * n_cls:(t + 1) * n_cls])
        return rows

    def pop(self, head: str, req_id: str, include_series: bool = False) -> dict | None:
        records = self._read(head, req_id)
        if records is None:
            return None
        path = _find_path(self.store_dir, head, req_id)
        if path is not None:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
        prompt_steps = [(r["n"], r["p"]) for r in records
                        if r.get("s") == "prefill" and "n" in r]
        decode_steps = [(r["n"], r["p"]) for r in records
                        if r.get("s") == "decode" and "n" in r]
        # R2: the engine writes a final aggregated record when the request
        # finishes -- pop is O(1) then (per-step lines only feed series)
        finals = [r for r in records if r.get("s") == "final"]
        if finals:
            f = finals[-1]
            result = {
                "prompt_scores": self._format(head, f.get("plast")),
                "response_scores_last": self._format(head, f.get("dlast")),
                "response_scores_max": self._format(head, f.get("dmax")),
            }
            if include_series:
                result["prompt_series"] = self._format_series(head, prompt_steps)
                result["response_series"] = self._format_series(head, decode_steps)
            return result
        result = {
            "prompt_scores": self._format(head, self._agg_last(head, prompt_steps)),
            "response_scores_last": self._format(head, self._agg_last(head, decode_steps)),
            "response_scores_max": self._format(head, self._agg_max(head, decode_steps)),
        }
        if include_series:
            result["prompt_series"] = self._format_series(head, prompt_steps)
            result["response_series"] = self._format_series(head, decode_steps)
        return result

    # -- aggregation -------------------------------------------------------
    def _agg_last(self, head: str, steps):
        """last scored token's flat class vector"""
        if not steps:
            return None
        n, flat = steps[-1]
        if n <= 0:
            return None
        rows = self.flat_to_rows(head, n, flat)
        return rows[-1]

    def _agg_max(self, head: str, steps):
        """elementwise max over every scored token -> flat class vector"""
        if not steps:
            return None
        mx = None
        for n, flat in steps:
            for row in self.flat_to_rows(head, n, flat):
                if mx is None:
                    mx = row
                else:
                    mx = [a if a >= b else b for a, b in zip(mx, row)]
        return mx

    def _format(self, head: str, vec):
        """flat class vector -> {task: {label: p}} via the layout offsets"""
        if vec is None:
            return None
        layout, _ = self.task_layout(head)
        _, id2label = self.schema(head)
        out = {}
        for task, start, stop in layout:
            labels = id2label.get(task, {})
            out[task] = {labels.get(str(i - start), str(i - start)): v
                         for i, v in zip(range(start, stop), vec[start:stop])}
        return out

    def _format_series(self, head: str, steps):
        if not steps:
            return None
        layout, _ = self.task_layout(head)
        _, id2label = self.schema(head)
        rows = []
        for n, flat in steps:
            rows.extend(self.flat_to_rows(head, n, flat))
        out = {}
        for task, start, stop in layout:
            labels = id2label.get(task, {})
            out[task] = [{labels.get(str(i - start), str(i - start)): row[i]
                          for i in range(start, stop)} for row in rows]
        return out

    def risk_flags(self, head: str, agg: dict, threshold: float) -> dict:
        heads, _ = self.schema(head)
        flags = {}
        for key in ("prompt_scores", "response_scores_max"):
            vecs = None
            if agg.get(key):
                vecs = []
                for (task, n_cls) in heads:
                    d = agg[key].get(task, {})
                    labels = self.schema(head)[1].get(task, {})
                    vecs.append([d.get(labels.get(str(i), str(i)), 0.0)
                                 for i in range(n_cls)])
            flags[key.replace("_scores", "_risk")] = self._risk(head, vecs, threshold)
        return flags

    def _risk(self, head: str, per_head_vec, threshold):
        if per_head_vec is None:
            return None
        heads, _ = self.schema(head)
        out = {}
        for (task, _n), vec in zip(heads, per_head_vec):
            # class 0 is always "normal"; risk = max prob of non-normal classes
            out[task] = round(max(vec[1:]) if len(vec) > 1 else 0.0, 6) >= threshold
        return out

    def pop_all(self, req_id: str) -> None:
        """Consume every head's score file for a request (error cleanup)."""
        for head in self.heads:
            self.pop(head, req_id)

    def cleanup_all(self) -> None:
        shutil.rmtree(self.store_dir, ignore_errors=True)
        os.makedirs(self.store_dir, exist_ok=True)


_shared_store = None


def get_store(head_dirs: dict | None = None):
    """Process-local singleton; both the engine-core and API processes build
    their own instance pointing at the same store directory."""
    global _shared_store
    if _shared_store is None:
        dirs = head_dirs or resolve_head_dirs()
        if not dirs:
            raise RuntimeError("no head dirs configured (PGUARD_HEADS/PGUARD_HEAD_PATH)")
        _shared_store = GuardStore(dirs)
    return _shared_store


if __name__ == "__main__":
    # in-module self-test: python -m pguard_vllm.store
    import tempfile

    tmp = tempfile.mkdtemp()
    schema = {
        "classifier": {"classification_heads": [["task_a", 2], ["task_b", 3]]},
        "task_schema": {
            "task_a": {"id2label": {"0": "normal", "1": "bad"}, "task_id": 0},
            "task_b": {"id2label": {"0": "normal", "1": "x", "2": "y"}, "task_id": 1},
        },
    }
    head_dirs = {}
    for name in ("lasttok", "ema"):
        hd = os.path.join(tmp, name)
        os.makedirs(hd, exist_ok=True)
        with open(os.path.join(hd, "config.json"), "w") as f:
            json.dump(schema, f)
        head_dirs[name] = hd
    tmp_store = os.path.join(tmp, "store")

    writer = GuardStore(head_dirs, store_dir=tmp_store)   # simulates engine core
    reader = GuardStore(head_dirs, store_dir=tmp_store)   # simulates api server

    # flat task-major vectors, exactly what the engine writes:
    # [task_a tok0 | task_a tok1 | task_b tok0 | task_b tok1]; per-token
    # values differ so layout scrambles cannot hide
    prefill_flat = [0.9, 0.1, 0.8, 0.1, 0.1]              # 1 token
    decode_flat = [0.3, 0.7, 0.1, 0.9,                    # task_a, tokens 0-1
                   0.2, 0.2, 0.6, 0.4, 0.1, 0.3]          # task_b, tokens 0-1
    for head in ("lasttok", "ema"):
        writer.append(head, "r1", prefill_flat, 1, is_decode=False)
        writer.append(head, "r1", decode_flat, 2, is_decode=True)
    assert reader.has_prefill("ema", "r1") and reader.has_prefill("lasttok", "r1")

    agg = reader.pop("ema", "r1", include_series=True)
    assert abs(agg["prompt_scores"]["task_a"]["bad"] - 0.1) < 1e-6
    assert abs(agg["response_scores_last"]["task_a"]["bad"] - 0.9) < 1e-6
    assert abs(agg["response_scores_last"]["task_b"]["y"] - 0.3) < 1e-6
    assert abs(agg["response_scores_max"]["task_a"]["bad"] - 0.9) < 1e-6
    assert abs(agg["response_scores_max"]["task_b"]["x"] - 0.2) < 1e-6
    assert abs(agg["response_scores_max"]["task_b"]["y"] - 0.6) < 1e-6
    assert len(agg["response_series"]["task_b"]) == 2     # two scored tokens
    risk = reader.risk_flags("ema", agg, 0.5)
    assert risk["prompt_risk"]["task_a"] is False
    assert risk["response_risk_max"]["task_a"] is True
    assert reader.pop("lasttok", "r1") is not None     # per-head files independent
    assert reader.pop("ema", "r1") is None             # consumed + unlinked

    # R2: final aggregated record -- engine writes it on finish; pop reads
    # this single line (O(1)) instead of replaying per-step records. The
    # vectors are full concatenated-class vectors (5 = 2 + 3 here).
    writer.append_final("ema", "r2", [0.4, 0.6, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.1, 0.2, 0.3],
                        [0.0, 0.0, 0.05, 0.9, 0.05], True, True)
    agg2 = reader.pop("ema", "r2")
    assert abs(agg2["prompt_scores"]["task_a"]["bad"] - 0.6) < 1e-6
    assert abs(agg2["response_scores_last"]["task_b"]["y"] - 0.3) < 1e-6
    assert abs(agg2["response_scores_max"]["task_b"]["x"] - 0.9) < 1e-6
    print("store self-test OK (multi-head, flat records, final record, cross-process)")
