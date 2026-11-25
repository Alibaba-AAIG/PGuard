import json, sys, time, threading
import urllib.request

BASE = sys.argv[1]
N_SEQ = int(sys.argv[2]) if len(sys.argv) > 2 else 8
MAX_TOKENS = 256

PROMPTS = [
    "用两段话介绍一下人工智能的发展历史",
    "Write a short product review for a wireless keyboard.",
    "帮我写一个Python函数，计算两个日期之间的工作日",
    "解释一下什么是知识蒸馏，以及它的优缺点",
    "Draft a polite follow-up email to a client who hasn't replied in two weeks.",
    "用简单语言向小学生解释为什么天是蓝的",
    "List 5 ideas for improving team meetings, with one sentence each.",
    "写一首关于江南水乡的现代诗",
] * ((N_SEQ + 7) // 8)
PROMPTS = PROMPTS[:N_SEQ]

def one(i, prompt, out, idx):
    body = json.dumps({"model": "pguard-backbone",
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": MAX_TOKENS, "temperature": 0.7}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as resp:
        r = json.loads(resp.read())
    dt = time.time() - t0
    ct = r["usage"]["completion_tokens"]
    out[idx] = (dt, ct)

def run_concurrent(n):
    out = [None] * n
    threads = [threading.Thread(target=one, args=(i, PROMPTS[i], out, i)) for i in range(n)]
    t0 = time.time()
    for t in threads: t.start()
    for t in threads: t.join()
    wall = time.time() - t0
    total_tokens = sum(x[1] for x in out)
    lat = [x[0] for x in out]
    print(f"  并发{n}: 总耗时={wall:.1f}s  完成token={total_tokens}  "
          f"吞吐={total_tokens/wall:.1f} tok/s  平均延迟={sum(lat)/len(lat):.1f}s  "
          f"最大延迟={max(lat):.1f}s")

def run_stream_ttft():
    body = json.dumps({"model": "pguard-backbone",
                       "messages": [{"role": "user", "content": "写一篇300字的关于低碳生活的短文"}],
                       "max_tokens": 300, "stream": True}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; tokens = 0
    with urllib.request.urlopen(req, timeout=600) as resp:
        prev = ""
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            c = json.loads(line[5:])
            if c["choices"][0].get("delta", {}).get("content"):
                if ttft is None:
                    ttft = time.time() - t0
                tokens += 1
    dt = time.time() - t0
    print(f"  流式: TTFT={ttft:.2f}s  生成{tokens}chunks/{dt:.1f}s = {tokens/dt:.1f} chunks/s")

print(f"== 目标 {BASE}  序列请求 x{N_SEQ} ==")
run_concurrent(N_SEQ)
run_stream_ttft()
