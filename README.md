# mvp-ecs-reranker

Self-hosted **cross-encoder rerankers** on ECS GPU (`g5.xlarge` / A10G) for deployment-oriented evaluation.

## Models

| Size | Model | Role |
| ---: | --- | --- |
| **22M** | `cross-encoder/ms-marco-MiniLM-L6-v2` | classic lightweight baseline |
| **~100–300M** | `jinaai/jina-reranker-v2-base-multilingual` | different family / architecture |
| **~335M** | `mixedbread-ai/mxbai-rerank-large-v1` | strong independent cross-encoder family |
| **568M** | `BAAI/bge-reranker-v2-m3` | strong established BGE baseline (**default**) |
| **~600M, optional** | `jinaai/jina-reranker-v3.5` | newer listwise reranking approach |

Swap request `"model"` to select among synced weights (one ECS task hosts all).

## Stack

| Piece | Value |
| --- | --- |
| Default model | `BAAI/bge-reranker-v2-m3` |
| Model weights | `s3://reranker-models-646821141010/` (all prefixes synced at boot) |
| Compute | 1 × `g5.xlarge` (1 × A10G) |
| Serving | sentence-transformers CrossEncoder + FastAPI |
| Region | `us-east-1` |
| CloudFormation stack | `ecs-reranker-mvp` |

## GitHub secrets

Repository secrets (Settings → Secrets and variables → Actions):

| Name | Purpose |
| --- | --- |
| `AWS_ACCESS_KEY_ID` | Deploy credentials |
| `AWS_SECRET_ACCESS_KEY` | Deploy credentials |
| `API_KEY` | Value required in `Authorization: Bearer` or `x-api-key` |

Optional variable: `AWS_REGION` (default `us-east-1`).

Set from a machine that already has AWS + `gh` auth:

```bash
./scripts/set-github-secrets.sh
# or manually:
gh secret set AWS_ACCESS_KEY_ID --body "$AWS_ACCESS_KEY_ID" --repo taixingbi/mvp-ecs-ranker
gh secret set AWS_SECRET_ACCESS_KEY --body "$AWS_SECRET_ACCESS_KEY" --repo taixingbi/mvp-ecs-ranker
gh secret set API_KEY --body "$API_KEY" --repo taixingbi/mvp-ecs-ranker
```

## Deploy

Push to `main` or run the **Deploy** workflow. Locally (Docker + AWS CLI required):

```bash
export API_KEY='your-shared-secret'
./scripts/deploy.sh
```

After deploy:

```bash
export SERVICE_URL=$(aws cloudformation describe-stacks \
  --region us-east-1 \
  --stack-name ecs-reranker-mvp \
  --query "Stacks[0].Outputs[?OutputKey=='ServiceUrl'].OutputValue" \
  --output text)
export API_KEY='your-shared-secret'

curl -sS -X POST "${SERVICE_URL}/v1/rerank" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d '{
    "model": "BAAI/bge-reranker-v2-m3",
    "query": "what is a panda?",
    "documents": [
      "The giant panda is a bear species endemic to China.",
      "Pandas is a Python library for data analysis.",
      "Bamboo is the primary food of giant pandas."
    ],
    "top_n": 2,
    "return_documents": true
  }'
```

Cold start includes S3 sync + model load; allow a few minutes before `/health` returns `{"status":"ok"}`.

## API

### Rerank (preferred)

`POST` `/v1/rerank` (also `/rerank` and `/`)

```json
{
  "model": "BAAI/bge-reranker-v2-m3",
  "query": "what is a panda?",
  "documents": ["doc A", "doc B"],
  "top_n": 2,
  "return_documents": true
}
```

Headers (either works):

- `Authorization: Bearer <API_KEY>`
- `x-api-key: <API_KEY>`

Success shape:

```json
{
  "model": "BAAI/bge-reranker-v2-m3",
  "results": [
    {"index": 0, "relevance_score": 0.98, "document": {"text": "..."}},
    {"index": 2, "relevance_score": 0.91, "document": {"text": "..."}}
  ],
  "usage": {"total_tokens": 0}
}
```

## Cost

A single on-demand `g5.xlarge` is roughly $1+/hour. Tear down when idle:

```bash
./scripts/destroy.sh
# keep ECR images:
DELETE_ECR=0 ./scripts/destroy.sh
```
# mvp-ecs-ranker
