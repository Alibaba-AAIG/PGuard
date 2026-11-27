# Multi-version acceptance matrix

Two runners, same acceptance steps:

- `run_matrix.sh` — one container per version via `VLLM_IMAGE` docker images
  (heavy: ~15-25 GB per image)
- `run_matrix_pip.sh` — pip-install each vLLM release inside the running
  container, restart, test, then restore the image's baseline at the end
  (light: wheels only, container is recreated from the image afterwards)

- `versions.py` — version -> docker image list for `run_matrix.sh`
- `run_matrix_pip.sh` — fetches the release list from PyPI (>= `MIN_VERSION`)
- `results/SUMMARY.md` / `results/SUMMARY-pip.md` — cumulative results

```bash
bash tests/versions/run_matrix.sh              # all versions in the list
bash tests/versions/run_matrix.sh 0.25.1       # subset
PULL=1 bash tests/versions/run_matrix.sh       # docker pull first

bash tests/versions/run_matrix_pip.sh          # pip sweep, all >= 0.25.0
bash tests/versions/run_matrix_pip.sh 0.26.0   # single version
PIP_INDEX_URL=https://pypi.org/simple \
  bash tests/versions/run_matrix_pip.sh        # other pip index (default: Tsinghua)
```

Slow pulls: prepend a registry mirror to the image tag, e.g.
`docker.m.daocloud.io/vllm/vllm-openai:vX.Y.Z`.
