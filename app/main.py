from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.api.routes.ingest import router as ingest_router
from app.api.routes.query import router as query_router
from app.api.routes.ai import router as ai_router
from app.container import get_llm_client, get_rag_pipeline, settings


app = FastAPI(title=settings.app_name, version=settings.app_version)
app.include_router(health_router)
app.include_router(ingest_router)
app.include_router(query_router)
app.include_router(ai_router)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "MP RAG backend ativo"}
