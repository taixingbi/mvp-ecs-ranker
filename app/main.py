import json
import os
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from sentence_transformers import CrossEncoder
from transformers import AutoModel

API_KEY = os.environ.get("API_KEY", "")
DEFAULT_MODEL_ID = os.environ.get("MODEL_ID", "BAAI/bge-reranker-v2-m3")
MODELS_ROOT = os.environ.get("MODELS_ROOT", "/models")

# HF model id -> local directory name under MODELS_ROOT
_DEFAULT_CATALOG: dict[str, str] = {
    "cross-encoder/ms-marco-MiniLM-L6-v2": "ms-marco-MiniLM-L6-v2",
    "jinaai/jina-reranker-v2-base-multilingual": "jina-reranker-v2-base-multilingual",
    "mixedbread-ai/mxbai-rerank-large-v1": "mxbai-rerank-large-v1",
    "BAAI/bge-reranker-v2-m3": "bge-reranker-v2-m3",
    "jinaai/jina-reranker-v3.5": "jina-reranker-v3.5",
}

# Listwise models expose model.rerank(query, documents); not CrossEncoder.
_LISTWISE_MODELS = frozenset({"jinaai/jina-reranker-v3.5"})


def _load_catalog() -> dict[str, str]:
    raw = os.environ.get("MODEL_CATALOG_JSON", "").strip()
    if raw:
        data = json.loads(raw)
        if not isinstance(data, dict) or not data:
            raise ValueError("MODEL_CATALOG_JSON must be a non-empty object")
        return {str(k): str(v) for k, v in data.items()}
    return {
        model_id: os.path.join(MODELS_ROOT, dirname)
        for model_id, dirname in _DEFAULT_CATALOG.items()
    }


MODEL_CATALOG = _load_catalog()

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "content-type,x-api-key,authorization",
    "Access-Control-Allow-Methods": "POST,OPTIONS",
}

app = FastAPI(title="mvp-ecs-reranker")
_models: dict[str, Any] = {}
_ready = False


def _json(status: int, body: dict[str, Any]) -> JSONResponse:
    return JSONResponse(status_code=status, content=body, headers=CORS_HEADERS)


def _authorized(request: Request) -> bool:
    if not API_KEY:
        return False
    if request.headers.get("x-api-key", "") == API_KEY:
        return True
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip() == API_KEY
    return False


def _available_models() -> list[str]:
    available: list[str] = []
    for model_id, path in MODEL_CATALOG.items():
        if os.path.isfile(os.path.join(path, "config.json")):
            available.append(model_id)
    return available


def _resolve_model_id(request_model: Any) -> str:
    if isinstance(request_model, str) and request_model.strip():
        model_id = request_model.strip()
    else:
        model_id = DEFAULT_MODEL_ID

    if model_id not in MODEL_CATALOG:
        raise ValueError(
            f"unknown model {model_id!r}; available: {_available_models()}"
        )
    path = MODEL_CATALOG[model_id]
    if not os.path.isfile(os.path.join(path, "config.json")):
        raise ValueError(f"model weights missing for {model_id!r} at {path}")
    return model_id


def _is_listwise(model_id: str) -> bool:
    return model_id in _LISTWISE_MODELS


def _get_model(model_id: str) -> Any:
    if model_id not in _models:
        path = MODEL_CATALOG[model_id]
        if _is_listwise(model_id):
            model = AutoModel.from_pretrained(
                path,
                trust_remote_code=True,
                dtype="auto",
            )
            model.eval()
            _models[model_id] = model
        else:
            _models[model_id] = CrossEncoder(path, trust_remote_code=True)
    return _models[model_id]


def _normalize_documents(payload: dict[str, Any]) -> list[str]:
    documents = payload.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError("documents must be a non-empty array")

    normalized: list[str] = []
    for item in documents:
        if isinstance(item, str):
            normalized.append(item)
        elif isinstance(item, dict) and isinstance(item.get("text"), str):
            normalized.append(item["text"])
        else:
            raise ValueError("each document must be a string or {text: string}")
    return normalized


def _parse_rerank(payload: dict[str, Any]) -> tuple[str, list[str], int | None]:
    query = payload.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    documents = _normalize_documents(payload)

    top_n = payload.get("top_n", payload.get("top_k"))
    if top_n is None:
        return query, documents, None
    if not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be a positive integer")
    return query, documents, min(top_n, len(documents))


def _rank_from_scores(
    scores: list[float], top_n: int | None
) -> list[dict[str, Any]]:
    ranked = sorted(
        (
            {"index": i, "relevance_score": float(score)}
            for i, score in enumerate(scores)
        ),
        key=lambda item: item["relevance_score"],
        reverse=True,
    )
    if top_n is not None:
        ranked = ranked[:top_n]
    return ranked


def _rerank(
    model_id: str, query: str, documents: list[str], top_n: int | None
) -> list[dict[str, Any]]:
    model = _get_model(model_id)

    if _is_listwise(model_id):
        kwargs: dict[str, Any] = {}
        if top_n is not None:
            kwargs["top_n"] = top_n
        raw = model.rerank(query, documents, **kwargs)
        results: list[dict[str, Any]] = []
        for item in raw:
            entry: dict[str, Any] = {
                "index": int(item["index"]),
                "relevance_score": float(item["relevance_score"]),
            }
            results.append(entry)
        return results

    pairs = [(query, doc) for doc in documents]
    scores = model.predict(pairs)
    return _rank_from_scores(list(scores), top_n)


@app.on_event("startup")
async def startup() -> None:
    global _ready
    available = _available_models()
    if not available:
        raise RuntimeError("no model weights found under MODEL_CATALOG paths")
    # Warm the default model so /health becomes ok after cold start.
    warm_id = DEFAULT_MODEL_ID if DEFAULT_MODEL_ID in available else available[0]
    _get_model(warm_id)
    _ready = True


@app.get("/health")
async def health() -> JSONResponse:
    if not _ready:
        return _json(503, {"status": "starting"})
    return _json(
        200,
        {
            "status": "ok",
            "default_model": DEFAULT_MODEL_ID,
            "available_models": _available_models(),
            "loaded_models": sorted(_models.keys()),
        },
    )


@app.get("/v1/models")
async def list_models() -> JSONResponse:
    return _json(
        200,
        {
            "object": "list",
            "data": [
                {"id": model_id, "object": "model", "owned_by": "local"}
                for model_id in _available_models()
            ],
        },
    )


@app.options("/")
@app.options("/rerank")
@app.options("/v1/rerank")
@app.options("/v1/models")
async def options() -> Response:
    return Response(status_code=204, headers=CORS_HEADERS)


@app.post("/")
@app.post("/rerank")
@app.post("/v1/rerank")
async def rerank(request: Request) -> JSONResponse:
    if not _authorized(request):
        return _json(401, {"error": "unauthorized"})

    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001
        return _json(400, {"error": "invalid JSON body"})

    try:
        model_id = _resolve_model_id(payload.get("model"))
        query, documents, top_n = _parse_rerank(payload)
    except ValueError as exc:
        return _json(400, {"error": str(exc)})

    try:
        results = _rerank(model_id, query, documents, top_n)
    except Exception as exc:  # noqa: BLE001
        return _json(500, {"error": "rerank failed", "detail": str(exc)})

    return_documents = bool(payload.get("return_documents"))
    if return_documents:
        for item in results:
            item["document"] = {"text": documents[item["index"]]}

    return _json(
        200,
        {
            "model": model_id,
            "results": results,
            "usage": {"total_tokens": 0},
        },
    )


@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def not_found(full_path: str) -> JSONResponse:
    return _json(404, {"error": "not found"})
