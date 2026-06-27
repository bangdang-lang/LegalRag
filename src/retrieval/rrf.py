from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


def reciprocal_rank_fusion(
    ranked_lists: Iterable[list[dict[str, Any]]],
    *,
    id_key: str = "chunk_id",
    k: int = 60,
    weights: list[float] | None = None,
) -> dict[str, float]:
    """
    Tính Reciprocal Rank Fusion:

        RRF(d) = sum_i weight_i / (k + rank_i(d))

    rank bắt đầu từ 1. RRF chỉ dùng thứ hạng, không phụ thuộc thang điểm
    khác nhau giữa BM25 và FAISS.
    """
    lists = list(ranked_lists)

    if weights is None:
        weights = [1.0] * len(lists)

    if len(weights) != len(lists):
        raise ValueError("Số lượng weights phải bằng số ranked_lists.")

    scores: dict[str, float] = defaultdict(float)

    for weight, results in zip(weights, lists):
        for rank, item in enumerate(results, start=1):
            doc_id = str(item[id_key])
            scores[doc_id] += float(weight) / (k + rank)

    return dict(scores)
