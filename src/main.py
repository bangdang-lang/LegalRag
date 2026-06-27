from pathlib import Path

from retrieval.hybrid_retriever import HybridRetriever
from graph_rerank import GraphExpander, GraphSearchPipeline
from evaluation_metrics.run_evaluation import run_evaluation
from evaluation_metrics.run_variant_report import main as run_variant_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]

print("Project root:", PROJECT_ROOT)


# =========================================================
# 1. Load hybrid retriever
# =========================================================

retriever = HybridRetriever(
    project_root=PROJECT_ROOT,
    model_name_or_path="models/embedding_model",

    bm25_dir="bm25_output",
    bm25_lookup_path="bm25_output/bm25_lookup.parquet",

    faiss_index_path="faiss_output/legal_chunks.faiss",
    embedding_metadata_path=(
        "embedding_output/embedding_metadata.parquet"
    ),
).load()


# =========================================================
# 2. Load graph
# =========================================================

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


# =========================================================
# 3. Create full search pipeline
# =========================================================

pipeline = GraphSearchPipeline(
    retriever=retriever,
    graph_expander=graph_expander,
)

# =========================================================
# 4. Run evaluation
# =========================================================

run_variant_report()