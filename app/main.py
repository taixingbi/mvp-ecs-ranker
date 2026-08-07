import os
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from sentence_transformers import CrossEncoder

API_KEY = os.environ.get("API_KEY", "")
MODEL_ID = os.environ.get("MODEL_ID", "BAAI/bge-reranker-v2-m3")
MODEL_PATH = os.environ.get("MODEL_PATH", f"/models/{MODEL_ID.split('/')[-1]}")

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "content-type,x-api-key,authorization",
    "Access-Control-Allow-Methods": "POST,OPTIONS",
}

app = FastAPI(title="mvp-ecs-reranker")
_model: CrossEncoder | None = None


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


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(MODEL_PATH)
    return _model


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


def _rerank(query: str, documents: list[str], top_n: int | None) -> list[dict[str, Any]]:
    model = _get_model()
    pairs = [(query, doc) for doc in documents]
    scores = model.predict(pairs)
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


@app.on_event("startup")
async def startup() -> None:
    _get_model()


@app.get("/health")
async def health() -> JSONResponse:
    if _model is None:
        return _json(503, {"status": "starting"})
    return _json(200, {"status": "ok", "model": MODEL_ID})


@app.options("/")
@app.options("/rerank")
@app.options("/v1/rerank")
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
        query, documents, top_n = _parse_rerank(payload)
    except ValueError as exc:
        return _json(400, {"error": str(exc)})

    request_model = payload.get("model")
    response_model = (
        request_model if isinstance(request_model, str) and request_model else MODEL_ID
    )

    try:
        results = _rerank(query, documents, top_n)
    except Exception as exc:  # noqa: BLE001
        return _json(500, {"error": "rerank failed", "detail": str(exc)})

    return_documents = bool(payload.get("return_documents"))
    if return_documents:
        for item in results:
            item["document"] = {"text": documents[item["index"]]}

    return _json(
        200,
        {
            "model": response_model,
            "results": results,
            "usage": {"total_tokens": 0},
        },
    )


@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def not_found(full_path: str) -> JSONResponse:
    return _json(404, {"error": "not found"})
