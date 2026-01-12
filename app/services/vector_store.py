from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np


@dataclass
class StoredDocument:
    document_id: str
    text: str
    metadata: Dict[str, Any]
    embedding: np.ndarray


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._store: List[StoredDocument] = []

    def add(self, documents: List[StoredDocument]) -> None:
        self._store.extend(documents)

    def similarity_search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Tuple[StoredDocument, float]]:
        if not self._store:
            return []
        scores: List[Tuple[StoredDocument, float]] = []
        for doc in self._store:
            score = float(np.dot(doc.embedding, query_embedding))
            scores.append((doc, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:top_k]
