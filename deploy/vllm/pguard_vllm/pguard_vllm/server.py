# PGuard-enabled vLLM API server (vLLM v0.34.0).
#
#   python -m pguard_vllm.server --model /models/backbone --port 13553
#
# Endpoints:
#   POST /v1/chat/completions  OpenAI-compatible chat; non-stream responses and
#                              the final stream chunk carry `guard_result`.
#   POST /v1/guard/score       standalone safety scoring (no generation).
#   GET  /v1/models /health
#
# Required env:
#   PGUARD_HEAD_PATH=/path/to/PGuard-<Head>-<Backbone>
#   VLLM_ENABLE_V1_MULTIPROCESSING=0   (forced here; guard store lives in-process)

import argparse
import inspect
import json
import logging
import os
import time
import uuid

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

import vllm  # noqa: E402  (imports trigger plugin registration + runner patch)
from vllm.engine.arg_utils import AsyncEngineArgs  # noqa: E402
from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm.v1.engine.async_llm import AsyncLLM  # noqa: E402

from .plugin import register as _register_plugin  # noqa: E402
from .store import GuardStore, get_store  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pguard_vllm.server")

_register_plugin()  # idempotent; also done via entry point

app = FastAPI(title="pguard-vllm")
STATE = {"engine": None, "tokenizer": None, "model_name": "pguard-backbone",
         "head_path": None, "threshold": 0.5,
         "verbose": bool(int(os.getenv("PGUARD_VERBOSE", "0"))),
         "interrupt": bool(int(os.getenv("PGUARD_INTERRUPT", "0")))}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _encode_messages(messages, add_generation_prompt: bool,
                     extra_kwargs: dict | None = None) -> list[int]:
    """chat-template encode; normalize transformers' return shapes to list[int].

    extra_kwargs forwards request-level chat_template_kwargs such as
    {"enable_thinking": false} for Qwen3(.5) thinking control.
    """
    encoded = STATE["tokenizer"].apply_chat_template(
        messages, add_generation_prompt=add_generation_prompt,
        **(extra_kwargs or {}))
    if hasattr(encoded, "input_ids"):        # BatchEncoding
        encoded = encoded["input_ids"]
    if isinstance(encoded, list) and encoded and isinstance(encoded[0], (list, tuple)):
        encoded = encoded[0]                 # batched -> first (only) row
    return [int(t) for t in encoded]


def build_guard_result(req_id: str, include_series: bool = False,
                       threshold: float | None = None,
                       heads: list[str] | None = None) -> dict | None:
    """Pop + aggregate the scores of every (requested) head for a request."""
    from .store import get_store
    store = get_store()
    threshold = threshold if threshold is not None else STATE["threshold"]
    heads = heads or store.heads
    results = {}
    for head in heads:
        agg = store.pop(head, req_id, include_series=include_series)
        if agg is None:
            continue
        agg["risk"] = store.risk_flags(head, agg, threshold)
        agg["threshold"] = threshold
        agg["head"] = head
        results[head] = agg
    store.clear_live(req_id)   # live status only serves the streaming check
    if not results:
        return None
    first = next(iter(results))
    return {
        "heads": list(results),
        "results": results,
        # backward-compatible top-level view of the first head
        "scores": results[first].get("prompt_scores"),
        "prompt_scores": results[first].get("prompt_scores"),
        "response_scores_last": results[first].get("response_scores_last"),
        "response_scores_max": results[first].get("response_scores_max"),
        "risk": results[first].get("risk"),
        "threshold": threshold,
        "head": results[first].get("head"),
    }


def sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _max_violation(head: str, task_scores: dict, threshold: float):
    """{task: {label: p}} -> (score, task, label) of the highest non-normal
    probability, or None if everything is below threshold."""
    heads, id2label = get_store().schema(head)
    best = None
    for task, labels in task_scores.items():
        for label, p in labels.items():
            if label == "normal":
                continue
            if p >= threshold and (best is None or p > best[0]):
                best = (p, task, label)
    return best


def _summarize_guard(store, heads: list[str], threshold: float,
                      agg_by_head: dict | None = None) -> dict:
    """Slim guard summary: the single most-likely violation across all online
    heads and both stages. This is what end users see by default."""
    best = None  # (score, task, label, stage, head)
    for head in heads:
        if agg_by_head is not None:
            agg = agg_by_head.get(head)
            if agg is None:
                continue
            candidates = [("prompt", agg.get("prompt_scores") or {}),
                          ("response", agg.get("response_scores_max") or {})]
            for stage, task_scores in candidates:
                hit = _max_violation(head, task_scores, threshold)
                if hit and (best is None or hit[0] > best[0]):
                    best = (hit[0], hit[1], hit[2], stage, head)
    if best is None:
        return {"risk": False, "category": None, "score": 0.0,
                "stage": None, "head": None, "threshold": threshold}
    score, task, label, stage, head = best
    return {"risk": True, "category": f"{task} / {label}",
            "score": round(score, 6), "stage": stage, "head": head,
            "threshold": threshold}


def _flat_task_scores(store, head: str, vec) -> dict:
    """Flat class vector (global class order) -> {task: {non-normal label: p}}."""
    _, id2label = store.schema(head)
    layout, _ = store.task_layout(head)
    task_scores = {}
    for task, start, stop in layout:
        labels = id2label.get(task, {})
        task_scores[task] = {
            labels.get(str(i - start), str(i - start)): vec[i]
            for i in range(start, stop)
            if labels.get(str(i - start), str(i - start)) != "normal"}
    return task_scores


def _latest_slim(store, heads: list[str], threshold: float,
                 req_id: str) -> dict | None:
    """Slim summary from a request's live status (streaming): per head the
    final prompt-side vector plus the running max over scored output tokens.
    The engine rewrites that one small file each step; reading it whole keeps
    a prompt-side hit visible no matter how many output records follow it."""
    live = store.read_live(req_id)
    if not live:
        return None
    best = None  # (score, task, label, stage, head)
    for head in heads:
        one = live.get(head)
        if not one:
            continue
        for stage, key in (("prompt", "plast"), ("response", "dmax")):
            vec = one.get(key)
            if not vec:
                continue
            hit = _max_violation(head, _flat_task_scores(store, head, vec),
                                 threshold)
            if hit and (best is None or hit[0] > best[0]):
                best = (hit[0], hit[1], hit[2], stage, head)
    if best is None:
        return None
    score, task, label, stage, head = best
    return {"risk": True, "category": f"{task} / {label}",
            "score": round(score, 6), "stage": stage, "head": head,
            "threshold": threshold}


def _finalize_guard(req_id: str, threshold: float, verbose: bool,
                    heads: list[str] | None = None) -> dict | None:
    """Pop all heads' scores for a finished request; slim by default."""
    from .store import get_store
    store = get_store()
    threshold = threshold if threshold is not None else STATE["threshold"]
    heads = heads or store.heads
    agg_by_head = {}
    for head in heads:
        agg = store.pop(head, req_id)
        if agg is not None:
            agg["head"] = head
            agg["risk"] = store.risk_flags(head, agg, threshold)
            agg_by_head[head] = agg
    store.clear_live(req_id)   # live status only serves the streaming check
    if not agg_by_head:
        return None
    slim = _summarize_guard(store, heads, threshold, agg_by_head)
    if verbose:
        slim["results"] = agg_by_head
        slim["heads"] = list(agg_by_head)
    return slim


# --------------------------------------------------------------------------
# /v1/chat/completions (hybrid mode)
# --------------------------------------------------------------------------
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body["messages"]
    stream = bool(body.get("stream", False))
    req_id = f"pguard-{uuid.uuid4().hex}"
    threshold = body.get("guard_threshold")
    if threshold is None:
        threshold = STATE["threshold"]
    verbose = bool(body.get("guard_verbose", STATE["verbose"]))
    interrupt = bool(body.get("guard_interrupt", STATE["interrupt"]))
    include_usage = bool((body.get("stream_options") or {}).get("include_usage"))

    prompt_ids = _encode_messages(messages, add_generation_prompt=True,
                                  extra_kwargs=body.get("chat_template_kwargs"))
    sampling = SamplingParams(
        temperature=float(body.get("temperature", 0.7)),
        top_p=float(body.get("top_p", 1.0)),
        max_tokens=int(body.get("max_tokens", 1024)),
        stop=body.get("stop") or None,
    )

    created = int(time.time())
    model_name = body.get("model") or STATE["model_name"]

    if not stream:
        final = None
        async for out in STATE["engine"].generate(
                {"prompt_token_ids": prompt_ids}, sampling, req_id):
            final = out
        text = final.outputs[0].text if final else ""
        resp = {
            "id": req_id, "object": "chat.completion", "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": len(prompt_ids),
                      "completion_tokens": len(final.outputs[0].token_ids) if final else 0},
        }
        guard = _finalize_guard(req_id, threshold, verbose)
        if guard is not None:
            resp["guard_result"] = guard
        return JSONResponse(resp)

    async def gen():
        prev = ""
        first = True
        aborted = None
        last_out = None
        store = get_store()
        try:
            async for out in STATE["engine"].generate(
                    {"prompt_token_ids": prompt_ids}, sampling, req_id):
                last_out = out
                # streaming guard: stop as soon as a category crosses the line
                if interrupt and aborted is None:
                    hit = _latest_slim(store, store.heads, threshold, req_id)
                    if hit is not None and hit["risk"]:
                        aborted = hit
                        await STATE["engine"].abort(req_id)
                        break
                delta = out.outputs[0].text
                piece = delta[len(prev):]
                prev = delta
                if first:
                    yield sse({"id": req_id, "object": "chat.completion.chunk",
                               "created": created, "model": model_name,
                               "choices": [{"index": 0,
                                            "delta": {"role": "assistant", "content": ""},
                                            "finish_reason": None}]})
                    first = False
                if piece:
                    yield sse({"id": req_id, "object": "chat.completion.chunk",
                               "created": created, "model": model_name,
                               "choices": [{"index": 0,
                                            "delta": {"content": piece},
                                            "finish_reason": None}]})
            guard = _finalize_guard(req_id, threshold, verbose)
            finish = "content_filter" if aborted else "stop"
            if first:  # aborted before anything was streamed
                yield sse({"id": req_id, "object": "chat.completion.chunk",
                           "created": created, "model": model_name,
                           "choices": [{"index": 0,
                                        "delta": {"role": "assistant", "content": ""},
                                        "finish_reason": None}]})
            yield sse({"id": req_id, "object": "chat.completion.chunk",
                       "created": created, "model": model_name,
                       "choices": [{"index": 0, "delta": {},
                                    "finish_reason": finish}],
                       **({"guard_result": guard} if guard else {})})
            if include_usage and last_out is not None:
                ct = len(last_out.outputs[0].token_ids)
                yield sse({"id": req_id, "object": "chat.completion.chunk",
                           "created": created, "model": model_name,
                           "choices": [],
                           "usage": {"prompt_tokens": len(prompt_ids),
                                     "completion_tokens": ct,
                                     "total_tokens": len(prompt_ids) + ct}})
            yield "data: [DONE]\n\n"
        except Exception:  # ensure the guard record never leaks
            store.pop_all(req_id)
            logger.exception("chat stream failed")

    return StreamingResponse(gen(), media_type="text/event-stream")


# --------------------------------------------------------------------------
# /v1/guard/score (standalone moderation)
# --------------------------------------------------------------------------
@app.post("/v1/guard/score")
async def guard_score(request: Request):
    body = await request.json()
    messages = body["messages"]
    return_series = bool(body.get("return_series", False))
    verbose = bool(body.get("guard_verbose", STATE["verbose"]))
    heads = body.get("heads") or None
    threshold = body.get("threshold")

    # PGuard convention: score points sit at the end of each text piece; an
    # empty assistant turn guarantees a prediction point after the prompt.
    marked = list(messages)
    if not marked or marked[-1].get("role") != "assistant":
        marked.append({"role": "assistant", "content": ""})

    prompt_ids = _encode_messages(marked, add_generation_prompt=False)
    req_id = f"pguard-{uuid.uuid4().hex}"
    sampling = SamplingParams(temperature=0.0, max_tokens=1)

    async for _ in STATE["engine"].generate(
            {"prompt_token_ids": prompt_ids}, sampling, req_id):
        pass

    result = build_guard_result(req_id, include_series=return_series,
                                threshold=threshold, heads=heads)
    if result is None:
        return JSONResponse({"error": "no guard scores captured "
                             "(prefix cache may have skipped all prefill tokens)"},
                            status_code=503)
    first = result["heads"][0]
    slim = _summarize_guard(get_store(), result["heads"], result["threshold"],
                            result["results"])
    slim["stage"] = "prompt"
    payload = {
        "object": "guard.score",
        "heads": result["heads"],
        # slim core conclusion (winning head/category across all heads)
        "risk": slim["risk"],
        "category": slim["category"],
        "score": slim["score"],
        "stage": slim["stage"],
        "head": slim["head"],
        "threshold": result["threshold"],
        "scores": result["results"][first]["prompt_scores"],
        "risk_detail": result["results"][first]["risk"]["prompt_risk"],
    }
    if verbose:
        payload["results"] = result["results"]
    if return_series:
        payload["series"] = {h: r.get("prompt_series")
                             for h, r in result["results"].items()}
    return JSONResponse(payload)


# --------------------------------------------------------------------------
# misc
# --------------------------------------------------------------------------
@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [
        {"id": STATE["model_name"], "object": "model", "owned_by": "pguard"}]}


@app.get("/health")
async def health():
    return {"status": "ok" if STATE["engine"] is not None else "starting"}


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def _ensure_plugin_installed() -> None:
    """vLLM inspects model architectures in a clean subprocess that only sees
    plugins via installed entry points; make sure our distribution is
    installed (editable, from the live-mounted source) before engine init."""
    from importlib.metadata import entry_points
    import subprocess
    import sys

    eps = [e for e in entry_points(group="vllm.general_plugins")
           if e.name == "pguard_vllm"]
    if eps:
        return
    src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-build-isolation",
         "--no-deps", "-q", "-e", src],
        check=True,
    )
    logger.info("pguard_vllm: installed editable distribution from %s", src)


def _self_test() -> int:
    """Streaming-interrupt check against a real store (no engine needed):
    python3 -m pguard_vllm.server --self-test"""
    import tempfile
    from .store import GuardStore

    tmp = tempfile.mkdtemp()
    schema = {
        "classifier": {"classification_heads": [["task_a", 2], ["task_b", 3]]},
        "task_schema": {
            "task_a": {"id2label": {"0": "normal", "1": "bad"}, "task_id": 0},
            "task_b": {"id2label": {"0": "normal", "1": "x", "2": "y"}, "task_id": 1},
        },
    }
    head_dir = os.path.join(tmp, "ema")
    os.makedirs(head_dir)
    with open(os.path.join(head_dir, "config.json"), "w") as f:
        json.dump(schema, f)
    store_dir = os.path.join(tmp, "store")
    engine = GuardStore({"ema": head_dir}, store_dir=store_dir)   # writer
    api = GuardStore({"ema": head_dir}, store_dir=store_dir)      # reader
    heads = ["ema"]
    benign = [0.9, 0.1, 0.8, 0.1, 0.1]

    def check(label, expect, req_id="r1"):
        hit = _latest_slim(api, heads, 0.5, req_id)
        got = None if hit is None else (hit["score"], hit["stage"], hit["category"])
        print("  %-44s -> %s" % (label, got if got == expect
                                 else "UNEXPECTED %r" % (got,)))
        return 0 if got == expect else 1

    bad = 0
    # risky prompt (task_a bad = 0.6) whose output tokens are all normal
    plast = [0.4, 0.6, 0.5, 0.3, 0.2]
    engine.append("ema", "r1", plast, 1, is_decode=False)
    engine.write_live("r1", {"ema": {"plast": plast, "dmax": None}})
    bad += check("prompt hit, nothing after it", (0.6, "prompt", "task_a / bad"))
    engine.append("ema", "r1", benign, 1, is_decode=True)
    bad += check("prompt hit + 1 normal output record",
                 (0.6, "prompt", "task_a / bad"))
    engine.append("ema", "r1", benign, 1, is_decode=True)
    bad += check("prompt hit + 2 normal output records",
                 (0.6, "prompt", "task_a / bad"))
    # a stronger output-side hit outranks the prompt one (task_b x = 0.9)
    engine.write_live("r1", {"ema": {"plast": plast,
                                     "dmax": [0.1, 0.0, 0.1, 0.9, 0.05]}})
    bad += check("prompt hit + stronger response hit",
                 (0.9, "response", "task_b / x"))
    engine.write_live("r1", {"ema": {"plast": benign, "dmax": benign}})
    bad += check("clean prompt + clean output", None)
    bad += check("request never scored", None, req_id="nope")
    api.pop_all("r1")
    bad += check("after pop_all", None)
    print("server self-test (streaming interrupt):",
          "FAILED (%d)" % bad if bad else "ok")
    return 1 if bad else 0


def main():
    parser = argparse.ArgumentParser(description="pguard-vllm server")
    parser.add_argument("--self-test", action="store_true",
                        help="run the streaming-interrupt self-test and exit")
    parser.add_argument("--model")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--served-model-name", default="pguard-backbone")
    parser.add_argument("--guard-threshold", type=float,
                        default=float(os.environ.get("PGUARD_THRESHOLD", "0.5")))
    parser.add_argument("--head-path", default=os.environ.get("PGUARD_HEAD_PATH"))
    args, remaining = parser.parse_known_args()
    if args.self_test:
        raise SystemExit(_self_test())
    if not args.model:
        parser.error("--model is required")
    _ensure_plugin_installed()
    from .store import get_store
    get_store().cleanup_all()   # drop stale score files from previous runs

    engine_args = AsyncEngineArgs(model=args.model)
    if remaining:
        # apply ONLY the options explicitly present in `remaining`; the
        # sub-parser's defaults must never clobber explicitly-set fields
        sub = argparse.ArgumentParser()
        AsyncEngineArgs.add_cli_args(sub)
        flag_to_dest = {opt: act.dest for act in sub._actions
                        for opt in act.option_strings}
        explicit = {flag_to_dest[tok] for tok in remaining if tok in flag_to_dest}
        extra = sub.parse_args(remaining)
        for dest in explicit:
            setattr(engine_args, dest, getattr(extra, dest))

    # CUDA-graph support: hooks capture mode requires eager (graph replay
    # skips Python hooks); PGUARD_CAPTURE_MODE=aux uses vLLM's EAGLE3-style
    # capture and can run with graphs (PGUARD_ENFORCE_EAGER=0). Default stays
    # eager (proven configuration).
    engine_args.enforce_eager = os.environ.get("PGUARD_ENFORCE_EAGER", "1") == "1"
    overrides = dict(engine_args.hf_overrides or {})
    overrides.update({"architectures": ["Qwen3_5PGuardForConditionalGeneration"]})
    engine_args.hf_overrides = overrides

    if not args.head_path:
        raise SystemExit("PGUARD_HEAD_PATH/--head-path is required")
    # startup fail-fast: every head checkpoint must be present and supported
    # before spending minutes on engine init
    from .store import resolve_head_dirs
    for _name, _dir in resolve_head_dirs().items():
        head_cfg_file = os.path.join(_dir, "config.json")
        if not os.path.isfile(head_cfg_file):
            raise SystemExit(f"head config not found: {head_cfg_file}")
        with open(head_cfg_file) as f:
            _head_meta = json.load(f)
        if _head_meta.get("pguard_version") not in ("lasttok", "ema", "gru"):
            raise SystemExit(f"unsupported head variant: "
                             f"{_head_meta.get('pguard_version')} "
                             f"(supported: lasttok, ema, gru)")
        if not os.path.isfile(os.path.join(_dir, "pytorch_model.bin")):
            raise SystemExit(f"head weights not found under: {_dir}")
    STATE["head_path"] = args.head_path
    STATE["threshold"] = args.guard_threshold
    STATE["model_name"] = args.served_model_name
    STATE["verbose"] = bool(int(os.getenv("PGUARD_VERBOSE", "0"))) or bool(int(os.getenv("PGUARD_DEBUG", "0")))
    STATE["interrupt"] = bool(int(os.getenv("PGUARD_INTERRUPT", "0")))
    STATE["engine_args"] = engine_args

    from transformers import AutoTokenizer
    STATE["tokenizer"] = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


@app.on_event("startup")
async def _startup():
    # created inside uvicorn's event loop: the engine's background tasks
    # (output handler) must bind to the same loop that serves HTTP
    engine = AsyncLLM.from_engine_args(STATE["engine_args"])
    if inspect.isawaitable(engine):
        engine = await engine
    STATE["engine"] = engine
    logger.info("pguard-vllm engine ready (head=%s)", STATE["head_path"])


if __name__ == "__main__":
    main()
