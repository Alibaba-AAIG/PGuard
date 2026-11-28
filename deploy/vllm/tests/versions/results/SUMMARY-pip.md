## pip run 2026-11-27 11:04:42 versions: 0.26.0
## pip run 2026-11-27 11:16:49 versions: 0.26.0
| 0.26.0 | qwen3_5 arch | SKIP |
| baseline | restore (0.25.1) | OK |
## pip run 2026-11-27 11:56:43 versions: 0.26.0
| 0.26.0 | healthy | FAIL (engine crash) |
| baseline | restore (0.25.1) | OK |
## pip run 2026-11-27 12:17:51 versions: 0.26.0
## pip run 2026-11-27 13:08:07 versions: 0.26.0
## pip run 2026-11-27 13:20:47 versions: 0.26.0
| 0.26.0 | healthy | FAIL (engine crash) |
## pip run 2026-11-27 13:28:53 versions: 0.26.0
| 0.26.0 | pip install | FAIL |
| 0.26.0 | restore | FAIL |
| baseline | restore | FAIL |
| 0.26.0 | compat | PASS (profile generic) |
| 0.26.0 | smoke | PASS (malicious 0.929, 与基线一致) |
| 0.26.0 | interrupt | PASS |
| 0.26.0 | cadence | PASS |
| 0.27.0 | torch 2.11.0 -> 2.13.0 (cu129 全栈) + torchvision 0.28.0+cu129 + FLASHINFER bypass | 前置 |
| 0.27.0 | compat | PASS (profile generic) |
| 0.27.0 | smoke | PASS (malicious 0.928, 与基线一致) |
| 0.27.0 | interrupt | PASS |
| 0.27.0 | cadence | PASS |
| 0.28.0 | compat | PASS (profile generic) |
| 0.28.0 | smoke | PASS (malicious 0.938, 与基线数值差 1% 内) |
| 0.28.0 | interrupt | PASS |
| 0.28.0 | cadence | PASS |
| 0.29.0 | compat | PASS (profile generic) |
| 0.29.0 | smoke | PASS (malicious 0.939, 与基线数值差 1% 内) |
| 0.29.0 | interrupt | PASS |
| 0.29.0 | cadence | PASS |
| 0.30.0 | compat | PASS (profile generic) |
| 0.30.0 | smoke | PASS (malicious 0.939, 与基线数值差 1% 内) |
| 0.30.0 | interrupt | PASS |
| 0.30.0 | cadence | PASS |
| 0.31.0 | 前置：依赖解析装齐（humming-kernels 等）、卸载遗留 torchaudio、flashinfer 家族对齐 0.7.x（jit-cache+sm89 手动装） | 前置 |
| 0.31.0 | compat | PASS (profile generic) |
| 0.31.0 | smoke | PASS (malicious 0.936, 与基线数值差 1% 内) |
| 0.31.0 | interrupt | PASS |
| 0.31.0 | cadence | PASS |
| 0.32.0 | 前置：humming-kernels 0.1.20、flashinfer-python 0.7.6 + jit-cache 0.7.6+cu129 (sm89)、卸载遗留 torchaudio（镜像 commit 方式绕开 crash 循环） | 前置 |
| 0.32.0 | compat | PASS (profile generic) |
| 0.32.0 | smoke | PASS (malicious 0.939, 与基线数值差 1% 内) |
| 0.32.0 | interrupt | PASS |
| 0.32.0 | cadence | PASS |
| 0.24.0 | compat | PASS (profile generic) |
| 0.24.0 | smoke | PASS (malicious 0.929, 与基线一致) |
| 0.24.0 | interrupt | PASS |
| 0.24.0 | cadence | PASS |
| 0.23.0 | compat | PASS (profile generic；补装 gguf) |
| 0.23.0 | smoke | PASS (malicious 0.931, 与基线数值差 1% 内) |
| 0.23.0 | interrupt | PASS |
| 0.23.0 | cadence | PASS |
| 0.22.0 | compat | PASS (profile generic；补装 gguf) |
| 0.22.0 | smoke | PASS (malicious 0.938, 与基线数值差 1% 内) |
| 0.22.0 | interrupt | PASS |
| 0.22.0 | cadence | PASS |
