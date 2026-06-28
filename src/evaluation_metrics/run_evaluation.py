from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluation_metrics.runner import EvaluationRunner
from graph_rerank import GraphExpander
from retrieval.hybrid_retriever import HybridRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.json"


def _resolve_path(
    value: str | Path,
) -> Path:
    path = Path(value)

    return (
        path
        if path.is_absolute()
        else PROJECT_ROOT / path
    )


def _first_value(
    value: Any,
    default: Any,
) -> Any:
    if value is None:
        return default

    if isinstance(value, list):
        return value[0] if value else default

    return value


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy config.json: "
            f"{CONFIG_PATH}"
        )

    return json.loads(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )


def run_evaluation() -> None:
    config = load_config()
    paths = config["paths"]
    evaluation = config["evaluation"]
    graph_config = config.get("graph", {})

    print("#" * 100)
    print(
        "LEGAL RAG EVALUATION: "
        "TUNE RRF/GRAPH, K CỐ ĐỊNH"
    )
    print(
        f"Project root: {PROJECT_ROOT}"
    )
    print("#" * 100)

    print()
    print("[1/4] Load retriever một lần")

    retriever = HybridRetriever(
        project_root=PROJECT_ROOT,
        model_name_or_path=(
            config["embedding_model"]["cache"]
        ),
        bm25_dir=paths["bm25_dir"],
        bm25_lookup_path=(
            paths["bm25_lookup_path"]
        ),
        faiss_index_path=(
            paths["faiss_index_path"]
        ),
        embedding_metadata_path=(
            paths[
                "embedding_metadata_path"
            ]
        ),
        device=evaluation.get(
            "device",
            "cpu",
        ),
    ).load()

    print()
    print("[2/4] Load graph một lần")

    initial_graph_weight = float(
        _first_value(
            graph_config.get(
                "graph_weight"
            ),
            0.3,
        )
    )

    graph_expander = GraphExpander(
        graph_path=_resolve_path(
            paths.get(
                "graph_dir",
                "graph",
            )
        ),
        retrieval_weight=(
            1.0 - initial_graph_weight
        ),
        graph_weight=(
            initial_graph_weight
        ),
        hop_decay=float(
            _first_value(
                graph_config.get(
                    "hop_decay"
                ),
                0.65,
            )
        ),
        direction=graph_config.get(
            "direction",
            "both",
        ),
        allowed_edge_types=None,
        edge_type_key=graph_config.get(
            "edge_type_key",
            "edge_type",
        ),
        edge_weight_key=graph_config.get(
            "edge_weight_key",
            "reference_count",
        ),
    ).load()

    print()
    print(
        "[3/4] Tune trên 300 train query"
    )

    runner = EvaluationRunner(
        retriever,
        graph_expander,
        config,
        PROJECT_ROOT,
    )

    best_params = runner.tune(
        _resolve_path(
            evaluation["train_path"]
        )
    )

    print()
    print(
        "[4/4] Test trên 100 query "
        "với K=5,10,50,100,200"
    )

    runner.evaluate_test(
        _resolve_path(
            evaluation["test_path"]
        ),
        best_params=best_params,
    )


if __name__ == "__main__":
    run_evaluation()
