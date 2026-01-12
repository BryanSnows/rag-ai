from app.core.config import get_settings
from app.services.embeddings import SimpleEmbedder
from app.services.llm import VitoriaLLMClient
from app.services.rag import RAGPipeline
from app.services.vector_store import InMemoryVectorStore
from app.services.vector_store_qdrant import QdrantVectorStore
from app.services.document_store import MongoDocumentStore

settings = get_settings()
_embedder = SimpleEmbedder(dim=settings.vector_dim)
if settings.qdrant_url:
    _vector_store = QdrantVectorStore(
        url=settings.qdrant_url,
        collection=settings.qdrant_collection,
        api_key=settings.qdrant_api_key,
        vector_dim=settings.vector_dim,
    )
else:
    _vector_store = InMemoryVectorStore()
_document_store = (
    MongoDocumentStore(
        uri=settings.mongo_uri,
        db=settings.mongo_db,
        collection=settings.mongo_collection,
    )
    if settings.mongo_uri
    else None
)
_llm_client = VitoriaLLMClient(
    model_name=settings.vitoria_model,
    api_key=settings.vitoria_api_key,
    base_url=settings.vitoria_base_url,
    temperature=settings.temperature,
)
_rag_pipeline = RAGPipeline(
    embedder=_embedder,
    vector_store=_vector_store,
    llm=_llm_client,
    document_store=_document_store,
    max_context_chars=settings.max_context_chars,
)


def get_rag_pipeline() -> RAGPipeline:
    return _rag_pipeline


def get_llm_client() -> VitoriaLLMClient:
    return _llm_client
