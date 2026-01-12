from fastapi import APIRouter, Depends

from app.models.schemas import QueryRequest, QueryResponse, SourceAttribution
from app.services.rag import RAGPipeline
from app.container import get_rag_pipeline

router = APIRouter(prefix="/query", tags=["consulta"])


@router.post("", response_model=QueryResponse)
def query(
    payload: QueryRequest,
    rag: RAGPipeline = Depends(get_rag_pipeline),
) -> QueryResponse:
    answer, docs, scores = rag.query(question=payload.question, top_k=payload.top_k)
    sources = [
        SourceAttribution(
            document_id=doc.document_id,
            score=round(score, 4),
            text=doc.text,
            metadata=doc.metadata,
        )
        for doc, score in zip(docs, scores)
    ]
    return QueryResponse(answer=answer, sources=sources)
