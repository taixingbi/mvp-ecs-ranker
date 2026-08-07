# Reranker models

| Size (weights) | Hugging Face ID | S3 prefix | Role | Serving |
| ---: | --- | --- | --- | --- |
| **~90MB** | `cross-encoder/ms-marco-MiniLM-L6-v2` | `ms-marco-MiniLM-L6-v2` | classic lightweight baseline | CrossEncoder / GPU |
| **~278M** | `jinaai/jina-reranker-v2-base-multilingual` | `jina-reranker-v2-base-multilingual` | different family / architecture | CrossEncoder / GPU |
| **~335M** | `mixedbread-ai/mxbai-rerank-large-v1` | `mxbai-rerank-large-v1` | strong independent cross-encoder | CrossEncoder / GPU |
| **~568M** | `BAAI/bge-reranker-v2-m3` | `bge-reranker-v2-m3` | strong BGE baseline (**default**) | CrossEncoder / GPU |

## Storage

- Bucket: `s3://reranker-models-646821141010/`
- Sync ships `model*.safetensors` plus tokenizer/config only (no ONNX / OpenVINO / cache)
