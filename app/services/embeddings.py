import numpy as np


class SimpleEmbedder:
    """
    Embedder determinístico baseado em hashing de tokens.
    Evita dependências pesadas e permite substituição por provedores reais.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    def _to_vector(self, tokens: list[str]) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in tokens:
            idx = hash(token) % self.dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        return vec if norm == 0 else vec / norm

    def embed(self, text: str) -> np.ndarray:
        return self._to_vector(self._tokenize(text))

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [self.embed(text) for text in texts]
