from pathlib import Path

from .hybrid_retriever import HybridRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_retrieval(query: str) -> None:
    retriever = HybridRetriever(
        project_root=PROJECT_ROOT,

        # Dùng thư mục model local nếu đã tải model vào project.
        # Ví dụ: PROJECT_ROOT / "models" / "Qwen3-Embedding-0.6B"
        model_name_or_path="models/Qwen3-Embedding-0.6B",

        bm25_dir="bm25_output",
        bm25_lookup_path="bm25_output/bm25_lookup.parquet",
        faiss_index_path="faiss_output/legal_chunks.faiss",
        embedding_metadata_path="embedding_output/embedding_metadata.parquet",
    ).load()

    results = retriever.search(
        query=query,
        top_k=10,
        bm25_top_k=30,
        faiss_top_k=30,
        rrf_k=60,
        bm25_weight=1.0,
        faiss_weight=1.0,
    )

    retriever.print_results(results)


if __name__ == "__main__":
    user_query = input("Nhập câu hỏi pháp luật: ").strip()
    run_retrieval(user_query)
