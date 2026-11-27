# Multi-version acceptance matrix

- `versions.py` — version -> docker image list (edit to add versions)
- `run_matrix.sh` — per version: pull (opt) -> compose up -> compat self-check
  -> smoke -> interrupt e2e -> cadence e2e; results in `results/`
- `results/SUMMARY.md` — one row per version/step

```bash
bash tests/versions/run_matrix.sh              # all versions in the list
bash tests/versions/run_matrix.sh 0.25.1       # subset
PULL=1 bash tests/versions/run_matrix.sh       # docker pull first
```

Slow pulls: prepend a registry mirror to the image tag, e.g.
`docker.m.daocloud.io/vllm/vllm-openai:vX.Y.Z`.
