from __future__ import annotations

import math
from collections.abc import Iterable

import pandas as pd


def evaluate_ranked_ids(
    ranked_ids: Iterable[str],
    relevant_ids: Iterable[str],
    k: int,
) -> dict[str, float]:
    if k <= 0:
        raise ValueError("K phải lớn hơn 0.")

    ranked = [str(item) for item in ranked_ids][:k]
    relevant = {str(item) for item in relevant_ids}

    hits = [
        1 if chunk_id in relevant else 0
        for chunk_id in ranked
    ]

    hit_count = sum(hits)

    first_relevant_rank = next(
        (
            rank
            for rank, chunk_id in enumerate(ranked, start=1)
            if chunk_id in relevant
        ),
        None,
    )

    dcg = sum(
        relevance / math.log2(rank + 1)
        for rank, relevance in enumerate(hits, start=1)
    )

    ideal_relevant_count = min(len(relevant), k)

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_relevant_count + 1)
    )

    return {
        "hit": float(hit_count > 0),
        "precision": hit_count / k,
        "recall": hit_count / len(relevant) if relevant else 0.0,
        "mrr": (
            1.0 / first_relevant_rank
            if first_relevant_rank is not None
            else 0.0
        ),
        "ndcg": dcg / idcg if idcg > 0 else 0.0,
    }


def summarize_metrics(
    per_query: pd.DataFrame,
    *,
    k: int,
) -> dict[str, float]:
    return {
        "K": int(k),
        "Hit@K": float(per_query["hit"].mean()),
        "Precision@K": float(per_query["precision"].mean()),
        "Recall@K": float(per_query["recall"].mean()),
        "MRR@K": float(per_query["mrr"].mean()),
        "nDCG@K": float(per_query["ndcg"].mean()),
    }


def build_metrics_table(
    summaries: list[dict[str, float]],
) -> pd.DataFrame:
    return pd.DataFrame(
        summaries,
        columns=[
            "K",
            "Hit@K",
            "Precision@K",
            "Recall@K",
            "MRR@K",
            "nDCG@K",
        ],
    )
