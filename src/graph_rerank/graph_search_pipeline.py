from __future__ import annotations

from typing import Any

from .graph_expander import GraphExpander
from .schemas import GraphSearchResult


class GraphSearchPipeline:
    """
    Hybrid retrieval -> RRF top-k -> graph expansion -> final top-k.
    """

    def __init__(
        self,
        retriever: Any,
        graph_expander: GraphExpander,
    ) -> None:
        self.retriever = retriever
        self.graph_expander = graph_expander

    def search(
        self,
        query: str,
        *,
        bm25_top_k: int = 50,
        faiss_top_k: int = 50,
        seed_top_k: int = 20,
        final_top_k: int = 10,
        rrf_k: int = 60,
        max_hops: int = 1,
        max_neighbors_per_node: int = 20,
    ) -> list[GraphSearchResult]:
        seeds = self.retriever.search(
            query=query,
            top_k=seed_top_k,
            bm25_top_k=bm25_top_k,
            faiss_top_k=faiss_top_k,
            rrf_k=rrf_k,
        )

        return self.graph_expander.expand(
            seeds,
            final_top_k=final_top_k,
            max_hops=max_hops,
            max_neighbors_per_node=max_neighbors_per_node,
            include_seeds=True,
        )

    @staticmethod
    def print_results(results: list[GraphSearchResult]) -> None:
        for rank, result in enumerate(results, start=1):
            print("=" * 100)
            print(f"TOP {rank}")
            print(f"chunk_id        : {result.chunk_id}")
            print(f"final_score     : {result.final_score:.8f}")
            print(f"retrieval_score : {result.retrieval_score:.8f}")
            print(f"graph_score     : {result.graph_score:.8f}")
            print(f"hop             : {result.hop}")
            print(f"source_seed_ids : {result.source_seed_ids}")

            if result.text:
                print("-" * 100)
                print(result.text)
