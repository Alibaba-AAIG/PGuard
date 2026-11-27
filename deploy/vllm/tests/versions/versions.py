# Version matrix for multi-version acceptance.
#
# Each entry maps a vLLM version to a docker image tag. Images must contain
# the qwen3_5 architecture (the first vLLM line that supports Qwen3.5-9B).
#
# Run (from the repo root):
#   bash tests/versions/run_matrix.sh              # every entry below
#   bash tests/versions/run_matrix.sh 0.25.1       # subset by version
#   PULL=1 bash tests/versions/run_matrix.sh       # docker pull before running

VERSIONS = [
    {"version": "0.25.1", "image": "vllm/vllm-openai:latest-cu129"},
    # {"version": "0.26.0", "image": "vllm/vllm-openai:v0.26.0"},
]

if __name__ == "__main__":
    for v in VERSIONS:
        print(v["version"], v["image"])
