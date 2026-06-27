from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RetrievalResult:
    """Một kết quả sau khi hợp nhất BM25 và FAISS bằng RRF."""

    chunk_id: str
    text: str
    rrf_score: float
    bm25_rank: int | None = None
    bm25_score: float | None = None
    faiss_rank: int | None = None
    faiss_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "rrf_score": self.rrf_score,
            "bm25_rank": self.bm25_rank,
            "bm25_score": self.bm25_score,
            "faiss_rank": self.faiss_rank,
            "faiss_score": self.faiss_score,
            "metadata": self.metadata,
        }
