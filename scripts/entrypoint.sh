#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-BAAI/bge-reranker-v2-m3}"
MODELS_ROOT="${MODELS_ROOT:-/models}"
MODELS_BUCKET="${MODELS_BUCKET:-s3://reranker-models-646821141010}"
ADAPTER_PORT="${ADAPTER_PORT:-8080}"
AWS_REGION="${AWS_REGION:-us-east-1}"

# Local dir names under MODELS_ROOT (must match app MODEL_CATALOG).
MODEL_DIRS=(
  ms-marco-MiniLM-L6-v2
  jina-reranker-v2-base-multilingual
  mxbai-rerank-large-v1
  bge-reranker-v2-m3
)

# Skip HF export duplicates / caches that blow disk on g5.xlarge root volume.
S3_SYNC_EXCLUDES=(
  --exclude ".cache/*"
  --exclude "onnx/*"
  --exclude "openvino/*"
  --exclude "assets/*"
  --exclude "*.onnx"
  --exclude "flax_model.*"
  --exclude "rust_model.*"
  --exclude "tf_model.*"
  --exclude "model.msgpack"
  --exclude "pytorch_model.bin"
  --exclude "pytorch_model.bin.index.json"
  --exclude "*.h5"
)

export AWS_DEFAULT_REGION="${AWS_REGION}"

mkdir -p "${MODELS_ROOT}"
for dir in "${MODEL_DIRS[@]}"; do
  src="${MODELS_BUCKET%/}/${dir}/"
  dst="${MODELS_ROOT%/}/${dir}"
  echo "Syncing ${src} -> ${dst}"
  mkdir -p "${dst}"
  aws s3 sync "${src}" "${dst}" --only-show-errors "${S3_SYNC_EXCLUDES[@]}"
  if [[ ! -f "${dst}/config.json" ]]; then
    echo "ERROR: config.json missing after S3 sync for ${dir}" >&2
    exit 1
  fi
  if ! compgen -G "${dst}/model*.safetensors" >/dev/null \
    && ! compgen -G "${dst}/pytorch_model*.bin" >/dev/null; then
    echo "ERROR: no model weights (*.safetensors) after S3 sync for ${dir}" >&2
    exit 1
  fi
done

echo "Starting reranker adapter on 0.0.0.0:${ADAPTER_PORT} (default=${MODEL_ID})"
export MODEL_ID MODELS_ROOT
exec python3 -m uvicorn app.main:app --host 0.0.0.0 --port "${ADAPTER_PORT}"
