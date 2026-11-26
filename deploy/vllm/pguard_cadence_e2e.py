"""End-to-end check of scoring cadence on the live guard service.

A long streaming request keeps decoding; shorter requests join while it runs.
Every late joiner must still carry its prompt-side Lasttok score, and only a
fraction of each request's output tokens may be scored.

Usage: python3 pguard_cadence_e2e.py [BASE] [N_JOINERS] [GAP_SECONDS]
The store snapshot reads the guard container (PGUARD_CONTAINER,
default pguard-vllm-guard) with docker exec.
"""
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:13553"
N_JOIN = int(sys.argv[2]) if len(sys.argv) > 2 else 16
GAP = float(sys.argv[3]) if len(sys.argv) > 3 else 0.35
HEADS = ("lasttok", "ema", "gru")   # response keys are the short head names


def post(body):
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=600)


def stream_long(a):
    body = {"model": "pguard-backbone",
            "messages": [{"role": "user",
                          "content": "写一篇600字的短文，介绍城市公共交通的发展历史"}],
            "max_tokens": 900, "stream": True, "guard_verbose": True}
    with post(body) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            c = json.loads(line[5:])
            a["id"] = c.get("id") or a["id"]
            if (c.get("choices") or [{}])[0].get("delta", {}).get("content"):
                a["chunks"] += 1
            if c.get("usage"):
                a["usage"] = c["usage"]
            if c.get("guard_result"):
                a["guard"] = c["guard_result"]
    a["done"] = True


def store_snapshot(req_id):
    """Per-head scored-token counts for one request's files, read in-container."""
    inner = r'''
import glob, json, sys
for f in sorted(glob.glob("/dev/shm/pguard_store/*__%s*.jsonl" % sys.argv[1])):
    dec = pre = 0
    for line in open(f):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("s") == "decode":
            dec += r.get("n", 1)
        elif r.get("s") == "prefill":
            pre += 1
    print("%s prefill_records=%d output_tokens_scored=%d"
          % (f.split("/")[-1].split("__")[0].replace("PGuard-", "").replace("-Qwen3.5-9B", ""), pre, dec))
'''
    out = subprocess.run(["docker", "exec",
                           os.environ.get("PGUARD_CONTAINER", "pguard-vllm-guard"),
                           "python3", "-c",
                          inner, req_id], capture_output=True, text=True)
    return out.stdout.strip() or out.stderr.strip()[-300:]


def main():
    a = {"id": None, "chunks": 0, "guard": None, "usage": None, "done": False}
    t = threading.Thread(target=stream_long, args=(a,), daemon=True)
    t.start()
    while a["chunks"] < 3:
        time.sleep(0.01)
    rows = []
    for i in range(N_JOIN):
        body = {"model": "pguard-backbone",
                "messages": [{"role": "user",
                              "content": "用一句话介绍第%d种常见水果的营养价值" % (i + 1)}],
                "max_tokens": 8, "guard_verbose": True}
        r = json.loads(post(body).read())
        res = (r.get("guard_result") or {}).get("results") or {}
        row = {"id": r.get("id"), "out_tokens": r["usage"]["completion_tokens"]}
        for h in HEADS:
            one = res.get(h) or {}
            row[h] = (one.get("prompt_scores") is not None,
                      one.get("response_scores_max") is not None)
        rows.append(row)
        time.sleep(GAP)
    snap_chunks = a["chunks"]
    snap = store_snapshot(a["id"]) if not a["done"] else "(long request already finished)"
    t.join()

    print("long request: id=%s output_tokens=%s" % (
        a["id"], (a["usage"] or {}).get("completion_tokens")))
    print("  store snapshot taken when %d content chunks (~tokens) had arrived:"
          % snap_chunks)
    for line in snap.splitlines():
        print("   ", line)
    bad = 0
    print("late joiners: %d (prompt arrives while the long request decodes)" % len(rows))
    for h in HEADS:
        name = h.replace("PGuard-", "").replace("-Qwen3.5-9B", "")
        p = sum(1 for x in rows if x[h][0])
        d = sum(1 for x in rows if x[h][1])
        print("  %-8s prompt scored: %2d/%d   requests with >=1 scored output token: %2d/%d"
              % (name, p, len(rows), d, len(rows)))
        bad += len(rows) - p
    outs = [x["out_tokens"] for x in rows]
    print("  joiner output tokens: total=%d (per request %d~%d)"
          % (sum(outs), min(outs), max(outs)))
    print("RESULT:", "ok" if bad == 0 else "FAILED (%d missing prompt scores)" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
