from app.core.config import get_settings
from app.services.embeddings import SimpleEmbedder
from app.services.llm import VitoriaLLMClient
from app.services.rag import RAGPipeline
from app.services.vector_store import InMemoryVectorStore

settings = get_settings()
_embedder = SimpleEmbedder(dim=settings.vector_dim)
_vector_store = InMemoryVectorStore()
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
    max_context_chars=settings.max_context_chars,
)


def get_rag_pipeline() -> RAGPipeline:
    return _rag_pipeline


def get_llm_client() -> VitoriaLLMClient:
    return _llm_client
