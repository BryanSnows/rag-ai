from typing import Any, Dict, List, Tuple
from uuid import uuid4

from .embeddings import SimpleEmbedder
from .llm import VitoriaLLMClient
from .vector_store import InMemoryVectorStore, StoredDocument


class RAGPipeline:
    def __init__(
        self,
        embedder: SimpleEmbedder,
        vector_store: InMemoryVectorStore,
        llm: VitoriaLLMClient,
        max_context_chars: int = 2000,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm
        self.max_context_chars = max_context_chars

    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        words = text.split()
        chunks: List[str] = []
        current: List[str] = []
        for word in words:
            current.append(word)
            if sum(len(w) + 1 for w in current) >= chunk_size:
                chunks.append(" ".join(current))
                current = []
        if current:
            chunks.append(" ".join(current))
        return chunks or [text]

    def ingest_document(
        self, content: str, metadata: Dict[str, Any] | None = None, source_id: str | None = None
    ) -> Tuple[str, int]:
        doc_id = source_id or str(uuid4())
        metadata = metadata or {}
        chunks = self._chunk_text(content)
        embeddings = self.embedder.embed_batch(chunks)
        stored = [
            StoredDocument(
                document_id=doc_id,
                text=chunk,
                metadata={**metadata, "chunk_index": idx},
                embedding=emb,
            )
            for idx, (chunk, emb) in enumerate(zip(chunks, embeddings))
        ]
        self.vector_store.add(stored)
        return doc_id, len(stored)

    def query(self, question: str, top_k: int = 3) -> Tuple[str, List[StoredDocument], List[float]]:
        query_emb = self.embedder.embed(question)
        results = self.vector_store.similarity_search(query_emb, top_k=top_k)
        documents = [doc for doc, _ in results]
        scores = [score for _, score in results]
        context_chunks = []
        total_len = 0
        for doc in documents:
            if total_len >= self.max_context_chars:
                break
            context_chunks.append(doc.text)
            total_len += len(doc.text)
        answer = self.llm.generate(question, context_chunks)
        return answer, documents, scores
