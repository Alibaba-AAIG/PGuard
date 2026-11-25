# pguard_vllm 补丁包设计（PGuard × vLLM v0.34.0）

> 目标：在 **vLLM v0.34.0 容器**（与 `vllm/vllm-openai:v0.34.0` 同版本源码核对过）内，
> 让 Qwen3.5-9B 在正常高并发生成的同时输出 PGuard 安全分，并提供独立安全审查接口。
> 原则：**不修改 vLLM 源码、不修改 PGuard 权重**；全部通过"继承 + 注册 + 运行时补丁"实现。

---

## 1. 已核实的 vLLM v0.34.0 内部事实（设计依据）

| # | 事实 | 出处（v0.34.0 源码） | 用途 |
|---|---|---|---|
| 1 | `ModelRegistry` 是模块级单例，`register_model(arch, cls_or_str)` 支持懒加载字符串 | `vllm/model_executor/models/registry.py:1660,1255` | 注册自定义架构 |
| 2 | `vllm.general_plugins` entry point 中的函数会被 `load_general_plugins()` 原样执行 | `vllm/plugins/__init__.py:77` | 插件入口做注册 |
| 3 | `hf_overrides` 字典会作为属性覆盖到 HF config | `vllm/config/model.py:362,640` | 把 architectures 换成自定义类名 |
| 4 | `AsyncLLM` 在 0.25.1/0.34.0 中**硬编码 MP client**：即使 `VLLM_ENABLE_V1_MULTIPROCESSING=0`，EngineCore（模型+钩子所在）也是独立进程 | `vllm/v1/engine/async_llm.py:146`（0.25.1 实测） | 打分经 /dev/shm 文件通道跨进程交给 API 层 |
| 5 | eager 路径下 `model_output = self.model(**model_inputs)`；FULL CUDA graph 模式是纯 replay | `vllm/v1/worker/gpu/model_runner.py:2164` | 必须 `--enforce-eager`，否则钩子不触发 |
| 6 | `SchedulerOutput.num_scheduled_tokens: dict[req_id, n]`，与 batch 行顺序一致，总和=真实 token 数 | `vllm/v1/core/sched/output.py:220` | **请求身份与分段**的核心 |
| 7 | `Qwen3NextModel.forward` 中 `hidden_states, residual = layer(...)`；HF `output_hidden_states[i]` ≙ `hidden + residual` | `vllm/model_executor/models/qwen3_next.py:723-760` | 层输出的正确取法 |
| 8 | `Qwen3_5ForConditionalGeneration.forward` 走 `self.language_model.model(...)`，层在 `self.language_model.model.layers` | `qwen3_5.py:555-583,340` | 钩子挂载位置 |
| 9 | PGuard 头输入 = HF `hidden_states` 元组按层索引堆叠 `(bs, layer, len, H)`，`selected_layers="20,24,28"` | `train/src/model/modeling_pguard_ind_wrapper.py:process_feature` | 特征构造与训练严格对齐 |
| 10 | `AsyncLLM.from_engine_args()` + `await engine.generate(prompt, sampling_params, request_id)` | `vllm/v1/engine/async_llm.py:280,740` | 自有 API 层驱动引擎 |

## 2. 总体架构

```
                    容器内（单进程：VLLM_ENABLE_V1_MULTIPROCESSING=0）
 ┌────────────────────────────────────────────────────────────────────┐
 │  FastAPI (pguard_vllm.server)                                      │
 │   /v1/chat/completions   混合模式：生成 + guard_result              │
 │   /v1/guard/score        独立审查：只打分不生成                     │
 │   /v1/models  /health                                              │
 │        │ AsyncLLM.generate()（OpenAI 风格流式 RequestOutput）       │
 │  ┌─────▼──────────────────────────────────────────────┐            │
 │  │ EngineCore (InprocClient) → GPUModelRunner         │            │
 │  │   execute_model()  ←── 运行时补丁：stash           │            │
 │  │      scheduler_output.num_scheduled_tokens         │            │
 │  │   Qwen3_5PGuardForConditionalGeneration (注册的新架构)           │
 │  │     ├─ 原版 Qwen3_5 前向（权重不变，FP8/bf16 均可） │            │
 │  │     ├─ layers[18,22,26] forward hooks ──► hidden+residual        │
 │  │     └─ PGuard 安全头（90MB，pg_weight checkpoint） │            │
 │  │         按 batch 分段逐请求打分 → GuardStore[req_id]│            │
 │  └────────────────────────────────────────────────────┘            │
 └────────────────────────────────────────────────────────────────────┘
```

加载链路：`vllm serve` 不用默认架构——通过 `--hf-overrides '{"architectures":["Qwen3_5PGuardForConditionalGeneration"]}'` + 插件注册，让 vLLM 用我们的子类加载**原版主干权重**；安全头从 `PGUARD_HEAD_PATH`（pg_weight 目录）独立加载。

工程注记（实测踩坑）：
- vLLM 会在**干净子进程**里 inspect 自定义架构（`python -m vllm.model_executor.models.registry`），插件必须以**已安装发行版的 entry point** 形式可见（服务启动时自动 `pip install --no-build-isolation --no-deps -e /opt/pguard_vllm`，源码目录 live-mount，改补丁无需重建镜像）；
- 补丁包内任何模块级名字都必须在 import 时可解析（曾因类型注解引用未导入的类导致 inspect 子进程 NameError）。

## 3. 高并发设计（关键点）

1. **不串行、不加锁**：钩子在每次 batch 前向内就地工作。一个 batch 里混着多个请求时，用 `num_scheduled_tokens`（req_id 顺序与 batch 行一致，总和=真实 token 数）把捕获到的 `(T, H)` 隐藏态切成段，段与请求一一对应，各自写各自的 `GuardStore[req_id]`。
2. **头计算批量做**：每步把整个 batch 的 token 一次性过安全头（90MB MLP，开销与官方 SGLang 集成同级，官方实测吞吐损失 0.8~2.3%）。
3. **prefill/decode 统一处理**：`CommonAttentionMetadata.is_prefilling` 区分阶段；prefill 段记为 prompt 侧分数（取每请求最后一条），decode 段逐 token 记为响应侧分数。
4. **enforce_eager 的取舍**：CUDA graph replay 不会执行 Python 钩子，因此 guard 模式必须 `--enforce-eager`（vLLM 的 continuous batching、PagedAttention、chunked prefill 等高并发能力全部保留，损失的主要是 graph 启动开销）。这也是官方 SGLang 集成需要 breakable-graph 补丁的同源问题，vLLM 侧用 eager 是最稳妥的等价选择。
5. **单进程（InprocClient）**：API 层与引擎同进程，`GuardStore` 是普通内存 dict，无跨进程 IPC；asyncio 单线程模型天然免锁。

## 4. 两种调用模式

### 4.1 混合模式：`POST /v1/chat/completions`
- 请求体 = 标准 OpenAI Chat 格式（messages / temperature / max_tokens / stream …）。
- 非流式：响应 JSON 在标准字段外追加 `guard_result`。
- 流式：每个 chunk 仍是标准 delta；**最后一个 data chunk** 追加 `guard_result` 字段，再发 `[DONE]`。
- `guard_result` 内容：
  ```json
  {
    "prompt_risk":  {"Crimes & Illegal Activities": {"normal": 0.97, "...": 0.01}, ...},
    "response_risk_max": {"...": 每个维度在生成 token 上的最大风险分布},
    "response_risk_last": {"...": 最后一个生成 token 的分布},
    "per_token": [[...], ...]   // 可选，stream 且 need_per_token 时逐 token
  }
  ```

### 4.2 独立审查：`POST /v1/guard/score`
- 请求：`{"messages": [...], "threshold": 0.5, "return_series": false}`；服务端按 PGuard 约定补空 assistant 轮，chat template 编码后**只做一次 prefill**（`max_tokens=1`），取 prompt 末位（ marker 位置）各类概率。
- 响应：`{"scores": {维度: {标签: 概率}}, "risk": {维度: 是否超阈值}, "max_risk": {...}}`，`return_series=true` 时附逐 token 序列（对齐官方 infer.py 输出格式）。

## 5. 补丁包结构

```
pguard_vllm/
├── pyproject.toml        # entry-point: vllm.general_plugins = pguard_vllm.plugin:register
├── pguard_vllm/
│   ├── plugin.py         # ModelRegistry.register_model("Qwen3_5PGuardForConditionalGeneration", 懒加载)
│   ├── head/             # 从 pguard-src/train/src/vendor 的安全头实现（lasttok 先行，ema/gru 预留）
│   ├── model.py          # Qwen3_5PGuardForConditionalGeneration：钩子 + 分段 + 头前向 + 写 GuardStore
│   ├── patch_runner.py   # GPUModelRunner.execute_model 补丁（stash num_scheduled_tokens）
│   ├── store.py          # GuardStore：按请求累积 prefill/decode 分数 + 聚合
│   └── server.py         # AsyncLLM 驱动的 FastAPI（两种模式 + OpenAI 兼容 chat）
```

环境变量：`PGUARD_HEAD_PATH`（checkpoint 目录）、`PGUARD_THRESHOLD`（默认 0.5）、`PGUARD_RETURN_SERIES`。

## 6. 限制与风险

1. **三种头均已移植，且支持多头同时在线**：`PGUARD_HEADS` 指定多个头，共享同一份主干与 union 捕获层（各头按 `feature_layers` 取自己的子集），每步并行评分；Lasttok（无状态）、EMA（每请求多尺度 EMA 状态，prefill 分块逐 token 折叠 + decode 每步更新）、GRU（query 池化摘要 h0 + 响应期 CfC 递推）。状态存于引擎进程的 `_seq_states[head]`，请求结束后按 store 文件消失（API 已 pop）自动清理，另有 4096 条 FIFO 上限兜底。
2. **`--enforce-eager` 必须**；不要开 speculative decode（MTP 与钩子叠加会破坏分段假设）。
3. 文本优先；多模态（图片）走 vLLM 多模态输入后，钩子仍按 token 分段工作，但未列入一期验证范围。
4. FP8 主干与 bf16 训练头之间存在精度差，分数可能整体偏移；上线前需用业务样本校准阈值。
5. chunked prefill 下 prompt 分数取"该请求最后一条 prefill 记录"，语义与官方 `PREFILL_RISK_MODE=last` 一致。
