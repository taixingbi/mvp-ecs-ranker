#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
ECR_REPO="${ECR_REPO:-mvp-ecs-reranker}"
STACK_NAME="${STACK_NAME:-ecs-reranker-mvp}"
IMAGE_TAG="${IMAGE_TAG:-$(git rev-parse --short HEAD 2>/dev/null || echo latest)}"
API_KEY="${API_KEY:-}"
MODEL_S3_URI="${MODEL_S3_URI:-s3://reranker-models-646821141010/bge-reranker-v2-m3/}"
MODEL_ID="${MODEL_ID:-BAAI/bge-reranker-v2-m3}"
MODEL_PATH="${MODEL_PATH:-/models/bge-reranker-v2-m3}"
DESIRED_COUNT="${DESIRED_COUNT:-1}"

if [[ -z "${API_KEY}" ]]; then
  echo "ERROR: set API_KEY" >&2
  exit 1
fi

IMAGE_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}"

echo "Ensuring ECR repository ${ECR_REPO} exists"
aws ecr describe-repositories --repository-names "${ECR_REPO}" --region "${AWS_REGION}" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "${ECR_REPO}" --region "${AWS_REGION}" >/dev/null

echo "Logging in to ECR"
aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

echo "Building ${IMAGE_URI}"
docker build -t "${IMAGE_URI}" .

echo "Pushing ${IMAGE_URI}"
docker push "${IMAGE_URI}"

echo "Deploying CloudFormation stack ${STACK_NAME}"
aws cloudformation deploy \
  --region "${AWS_REGION}" \
  --stack-name "${STACK_NAME}" \
  --template-file infra/ecs-stack.yaml \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
    "ApiKey=${API_KEY}" \
    "ImageUri=${IMAGE_URI}" \
    "ModelS3Uri=${MODEL_S3_URI}" \
    "ModelId=${MODEL_ID}" \
    "ModelPath=${MODEL_PATH}" \
    "DesiredCount=${DESIRED_COUNT}" \
  --no-fail-on-empty-changeset

SERVICE_URL="$(aws cloudformation describe-stacks \
  --region "${AWS_REGION}" \
  --stack-name "${STACK_NAME}" \
  --query "Stacks[0].Outputs[?OutputKey=='ServiceUrl'].OutputValue" \
  --output text)"

echo "Deployed. ServiceUrl=${SERVICE_URL}"
echo "Example:"
echo "  curl -sS -X POST \"${SERVICE_URL}/v1/rerank\" \\"
echo "    -H 'content-type: application/json' \\"
echo "    -H \"Authorization: Bearer \${API_KEY}\" \\"
echo "    -d '{\"model\":\"${MODEL_ID}\",\"query\":\"what is pandas?\",\"documents\":[\"Pandas is a Python library.\",\"The panda is a bear.\"],\"top_n\":2}'"
