from typing import List, Tuple
from uuid import uuid4

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from .vector_store import StoredDocument


class QdrantVectorStore:
    def __init__(
        self,
        url: str,
        collection: str,
        api_key: str | None,
        vector_dim: int,
    ) -> None:
        self.collection = collection
        self.client = QdrantClient(url=url, api_key=api_key, prefer_grpc=False)
        self._ensure_collection(vector_dim)

    def _ensure_collection(self, dim: int) -> None:
        try:
            self.client.get_collection(self.collection)
        except Exception:
            self.client.recreate_collection(
                collection_name=self.collection,
                vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
            )

    def add(self, documents: List[StoredDocument]) -> None:
        if not documents:
            return
        payloads = []
        vectors = []
        ids: List[str] = []
        for doc in documents:
            ids.append(str(uuid4()))  # Qdrant IDs must be UUID or int
            payloads.append({"text": doc.text, "document_id": doc.document_id, **doc.metadata})
            vectors.append(doc.embedding.astype(float).tolist())
        self.client.upsert(
            collection_name=self.collection,
            points=qmodels.Batch(ids=ids, vectors=vectors, payloads=payloads),
        )

    def similarity_search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Tuple[StoredDocument, float]]:
        search_result = self.client.search(
            collection_name=self.collection,
            query_vector=query_embedding.astype(float).tolist(),
            limit=top_k,
            with_payload=True,
        )
        docs: List[Tuple[StoredDocument, float]] = []
        for point in search_result:
            payload = point.payload or {}
            text = payload.pop("text", "")
            doc_id = payload.pop("document_id", None) or str(point.id)
            chunk_index = payload.get("chunk_index", 0)
            doc = StoredDocument(
                document_id=str(doc_id),
                text=text,
                metadata={"chunk_index": chunk_index, **payload},
                embedding=query_embedding,  # not used downstream
            )
            docs.append((doc, float(point.score)))
        return docs
