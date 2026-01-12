from fastapi import APIRouter, Depends

from app.models.schemas import DocumentIngestRequest, DocumentIngestResponse
from app.services.rag import RAGPipeline
from app.container import get_rag_pipeline

router = APIRouter(prefix="/ingest", tags=["ingestao"])


@router.post("", response_model=DocumentIngestResponse)
def ingest_document(
    payload: DocumentIngestRequest,
    rag: RAGPipeline = Depends(get_rag_pipeline),
) -> DocumentIngestResponse:
    document_id, segments = rag.ingest_document(
        content=payload.content,
        metadata=payload.metadata,
        source_id=payload.source_id,
    )
    return DocumentIngestResponse(document_id=document_id, segments=segments)
