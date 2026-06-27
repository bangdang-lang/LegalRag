from pathlib import Path

from retrieval.hybrid_retriever import HybridRetriever
from graph_rerank import GraphExpander, GraphSearchPipeline
from models.llm_answer_generator import LLMAnswerGenerator


PROJECT_ROOT = Path(__file__).resolve().parents[1]

print("Project root:", PROJECT_ROOT)


retriever = HybridRetriever(
    project_root=PROJECT_ROOT,
    model_name_or_path="models/embedding_model",
    bm25_dir="bm25_output",
    bm25_lookup_path="bm25_output/bm25_lookup.parquet",
    faiss_index_path="faiss_output/legal_chunks.faiss",
    embedding_metadata_path="embedding_output/embedding_metadata.parquet",
).load()

graph_expander = GraphExpander(
    graph_path=PROJECT_ROOT / "graph",
    retrieval_weight=0.70,
    graph_weight=0.30,
    hop_decay=0.65,
    direction="both",
    allowed_edge_types=None,
    edge_type_key="type",
    edge_weight_key="weight",
).load()

pipeline = GraphSearchPipeline(
    retriever=retriever,
    graph_expander=graph_expander,
)


query = input("Nhập câu hỏi pháp luật: ").strip()

results = pipeline.search(
    query=query,
    bm25_top_k=50,
    faiss_top_k=50,
    seed_top_k=20,
    final_top_k=10,
    rrf_k=60,
    max_hops=1,
    max_neighbors_per_node=20,
)

pipeline.print_results(results)

llm_generator = LLMAnswerGenerator(
    model_name_or_path=(
        PROJECT_ROOT
        / "models"
        / "llm_models"
        / "Qwen3"
    ),
    device=None,
    max_context_chars=18000,
    max_new_tokens=512,
    do_sample=False,
    local_files_only=True,
).load()

answer = llm_generator.generate_answer(
    query=query,
    results=results,
)

print("\n" + "=" * 100)
print("CÂU TRẢ LỜI")
print("=" * 100)
print(answer)