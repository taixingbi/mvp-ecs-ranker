#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-BAAI/bge-reranker-v2-m3}"
MODEL_PATH="${MODEL_PATH:-/models/bge-reranker-v2-m3}"
MODEL_S3_URI="${MODEL_S3_URI:-s3://bedrock-models-646821141010/rerankers/bge-reranker-v2-m3/}"
ADAPTER_PORT="${ADAPTER_PORT:-8080}"
AWS_REGION="${AWS_REGION:-us-east-1}"

export AWS_DEFAULT_REGION="${AWS_REGION}"

echo "Syncing model from ${MODEL_S3_URI} -> ${MODEL_PATH}"
mkdir -p "${MODEL_PATH}"
aws s3 sync "${MODEL_S3_URI}" "${MODEL_PATH}" --only-show-errors

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "ERROR: config.json missing after S3 sync. Check MODEL_S3_URI=${MODEL_S3_URI}" >&2
  exit 1
fi

echo "Starting reranker adapter on 0.0.0.0:${ADAPTER_PORT} (model=${MODEL_ID})"
export MODEL_ID MODEL_PATH
exec python3 -m uvicorn app.main:app --host 0.0.0.0 --port "${ADAPTER_PORT}"
