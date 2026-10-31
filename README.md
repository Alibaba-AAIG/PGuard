# PGuard: A Native, Streaming, and Plug-in Guardrail for Multimodal Large Language Models

**Alibaba AAIG**

[🤗 Hugging Face](https://huggingface.co/collections/Alibaba-AAIG/pguard) · [🤖 ModelScope](https://modelscope.cn/collections/Alibaba-AAIG/PGuard) · [Training](train/README.md) · [Deployment](deploy/README.md) · [Technical Report](#citation)

## Introduction

PGuard is a native, streaming, and plug-in guardrail built on a frozen foundation model backbone. It directly leverages the backbone’s intermediate representations for token-level safety detection, without altering its original inference pathway or generation behavior. Evaluations across more than 30 benchmarks spanning text and multimodal safety, jailbreaks, cyber and CBRN risks, multilingual scenarios, and real-world traffic demonstrate competitive detection quality with limited overhead on large-scale model serving.

### Key Features

- **Adaptive Safety Representations:** Dynamic layer gating selectively aggregates residual-stream representations across layers to extract safety-relevant signals during both prefill and autoregressive decoding.
- **Flexible Modeling Architectures:** A lightweight stateless variant and sequence-aware variants built on multi-scale causal EMA-attention and GRU-based recurrent modules offer different trade-offs between detection quality and computational cost.
- **Low Serving Overhead:** SGLang integration aligns risk computation and reporting with the engine’s scheduling mechanisms, minimizing redundant computation and synchronization overhead without modifying the inference engine’s codebase.
- **Configurable Streaming Decisions:** Per-token risk scores support deployment-time configuration of category-specific thresholds and enforcement actions as responses unfold.

## How It Works

```text
Text / image-text conversation
              │
              ▼
       Frozen backbone ───────────────────────► Generated tokens
              ▼                                      │
       Selected hidden states                        │
              ▼                                      │
       Dynamic layer gating                          │
              ▼                                      │
   Stateless / EMA-attention / GRU                   │
              ▼                                      │
       Task-specific classifiers                     │
              │                                      │
              └─────► Risk scores + token stream ◄───┘
                                  │
                                  ▼
                     Application safety policy
```

The guard head produces safety signals. Threshold selection and enforcement actions, such as passing, interrupting, or escalating a response for review, belong to the application.

| Head | Training `--version` | Serving `PGUARD_MODELING_VARIANT` | Representation modeling |
|---|---|---|---|
| Stateless | `lasttok` | `stateless` | Classifies each token position without carrying guard state between steps |
| EMA-attention | `ema` | `seq` | Combines multi-scale causal EMA summaries with attention |
| GRU | `gru` | `cfc` | Summarizes the prompt and maintains recurrent state during response generation |

## Evaluation

PGuard achieves the following benchmark-average F1 scores on **Qwen3.5-9B**:

| Variant | Text queries | Text responses | Multimodal queries | Multimodal responses |
|---|---:|---:|---:|---:|
| PGuard-lasttok | 0.822 | 0.795 | 0.635 | 0.644 |
| PGuard-ema | **0.836** | 0.833 | **0.741** | 0.772 |
| PGuard-gru | **0.836** | **0.860** | 0.735 | **0.798** |

On Qwen3.5-122B-A10B, PGuard incurs **0.82–2.26% output throughput loss** in the tested SGLang configurations. See the [technical report](#citation) for full results and evaluation settings.

## Supported Backbones

Training and serving support are listed separately.

| Model family | Input | Training / offline inference | SGLang serving in this repository |
|---|---|---|---|
| Qwen3 | Text | Supported | — |
| Qwen3-VL | Image-text | Supported | — |
| Qwen3.5 | Image-text | Supported | 9B, 35B-A3B, 122B-A10B |
| GLM-4.5V | Image-text | Supported | Supported |
| DeepSeek-V4-Flash | Text | Supported | Supported |

## Quick Start

- **[Training and Offline Inference](train/README.md):** Installation, data preparation, head training, and batch scoring.
- **[SGLang Deployment](deploy/README.md):** Installation, model serving, and streaming risk scores.

## Safety Taxonomy

| Risk dimension | Risk categories |
|---|---|
| Crimes & Illegal Activities | Pornographic Contraband · Drug Crimes · Dangerous Weapons · Property Infringement · Economic Crimes |
| Hate Speech | Abusive Curses · Defamation · Threats & Intimidation · Cyberbullying |
| Physical & Mental Health | Personal Safety · Mental Health |
| Ethics & Morality | Social Ethics · Science & Tech Ethics |
| Data Privacy | Personal Privacy · Commercial Secret |
| Cybersecurity | Access Control · Malicious Code · Hacker Attack · Physical Security · Social Engineering, Phishing |
| Extremism | Violent Terrorist Activities · Social Disruption · Extremist Ideological Trends |
| Inappropriate Suggestions | Finance · Medicine · Law |
| Risks Involving Minors | Corruption of Minors · Minor Abuse & Exploitation · Minor Delinquency |

## Repository Structure

```text
.
├── train/                 # Head training and offline inference
│   ├── train.py
│   ├── infer.py
│   ├── examples/
│   └── src/
├── deploy/                # SGLang integration
│   ├── pyproject.toml
│   └── src/pguard/
└── LICENSE
```

## Citation

```bibtex
@techreport{aaig2026pguard,
  title       = {PGuard: A Native, Streaming, and Plug-in Guardrail for Multimodal Large Language Models},
  author      = {{Alibaba AAIG}},
  institution = {Alibaba AAIG},
  year        = {2026},
  month       = oct
}
```

## License

This repository is licensed under the [Apache License 2.0](LICENSE).
