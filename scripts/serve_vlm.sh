#!/usr/bin/env bash
# Serve Qwen3.8-27B (thinking VLM) behind an OpenAI-compatible API with tool calling, in the
# vllm-q38 env. bf16 weights are ~55 GB: TP=4 on A100-40GB (configs/default.yaml: compute.vlm_gpus).
# Flags follow https://recipes.vllm.ai/Qwen/Qwen3.8-27B (reasoning parser qwen3, tool parser qwen3_coder).
# Binds to all interfaces so other users on the server (and the lab network) can use it: see
# docs/VLM_SERVICE.md. No API key: the server is reachable by anyone who can reach this host.
set -euo pipefail
MODEL=${MODEL:-Qwen/Qwen3.8-27B}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
TP=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
# Speculative decoding with the model's own MTP layer (the recipe's setting): lossless, faster decoding, which is most of
# a thinking request. SPEC= (empty) turns it off.
DEFAULT_SPEC='{"method":"mtp","num_speculative_tokens":3}'
SPEC=${SPEC-$DEFAULT_SPEC}
EXTRA=()
if [[ -n "$SPEC" ]]; then EXTRA+=(--speculative-config "$SPEC"); fi
exec /home/gochen/miniconda3/envs/vllm-q38/bin/vllm serve "$MODEL" \
  "${EXTRA[@]}" \
  --tensor-parallel-size "$TP" \
  --max-model-len 131072 \
  --limit-mm-per-prompt '{"image": 24}' \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --max-logprobs 20 \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-8000}"
