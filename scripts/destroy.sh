#!/usr/bin/env bash
# Destroy the ECS reranker MVP (CloudFormation stack + optional ECR repo).
# Usage:
#   ./scripts/destroy.sh
#   DELETE_ECR=0 ./scripts/destroy.sh   # keep ECR images
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${ROOT_DIR}/scripts/rm-ecs.sh" "$@"
