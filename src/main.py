from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from core.config import AppConfig
from evaluation_metrics.runner import EvaluationRunner
from graph_rerank.graph_builder import LegalHierarchyGraphBuilder
from graph_rerank.graph_search_pipeline import GraphSearchPipeline
from models.answer_generator import LLMAnswerGenerator
from models.embedder import EmbeddingBuilder
from models.index_builders import BM25IndexBuilder, FaissIndexBuilder
from models.model_downloader import ModelDownloader
from utils.logger import configure_logging


class LegalRAGApplication:
    """Build indexes, tune hyperparameters and compare retrieval methods."""

    def __init__(self, config_path: str | Path = "config.json") -> None:
        self.config = AppConfig.load(config_path)
        configure_logging(self.config.get("project.log_level", "INFO"))
        self.logger = logging.getLogger(self.__class__.__name__)

    def download_models(self) -> None:
        downloader = ModelDownloader(self.config)
        downloader.download_embedding_model()
        if bool(self.config.get("generation.enabled", False)):
            downloader.download_generation_model()

    def build(self) -> None:
        actions = {
            "download_models": self.download_models,
            "embed": lambda: EmbeddingBuilder(self.config).build(),
            "faiss": lambda: FaissIndexBuilder(self.config).build(),
            "bm25": lambda: BM25IndexBuilder(self.config).build(),
            "graph": lambda: LegalHierarchyGraphBuilder(self.config).build(),
        }
        steps = list(self.config.get("pipeline.build_steps", actions.keys()))
        for index, step in enumerate(steps, start=1):
            if step not in actions:
                raise ValueError(f"Unknown build step: {step}. Supported: {list(actions)}")
            self.logger.info("[BUILD %d/%d] %s", index, len(steps), step)
            actions[step]()

    def train_tune_evaluate(self) -> dict[str, Any]:
        """Tune on 300 queries and print the final five-method comparison table."""
        if bool(self.config.get("pipeline.rebuild_before_evaluation", False)):
            self.build()
        return EvaluationRunner(self.config).run_experiment()

    def experiment(self) -> dict[str, Any]:
        return self.train_tune_evaluate()

    def evaluate(self) -> dict[str, Any]:
        return self.train_tune_evaluate()

    def search(self, query: str | None = None, method: str = "hybrid+graph") -> list[dict[str, Any]]:
        query = query or self.config.get("pipeline.query")
        if not query:
            raise ValueError("A query is required. Pass --query or set pipeline.query in config.json.")
        top_k = int(self.config.get("retrieval.final_top_k", 200))
        results = GraphSearchPipeline(self.config).load().search_method(query, method=method, top_k=top_k)
        output = [
            {"rank": rank, "chunk_id": item.chunk_id, "score": item.score, "source": item.source, "text": item.text}
            for rank, item in enumerate(results, start=1)
        ]
        for item in output[:20]:
            print(item)
        return output

    def answer(self, query: str | None = None) -> str:
        query = query or self.config.get("pipeline.query")
        if not query:
            raise ValueError("A query is required. Pass --query or set pipeline.query in config.json.")
        results = GraphSearchPipeline(self.config).load().search(query)
        answer = LLMAnswerGenerator(self.config).generate(query, results)
        print(answer)
        return answer

    def run(self, mode: str | None = None, query: str | None = None, method: str = "hybrid+graph") -> Any:
        selected = mode or self.config.get("pipeline.mode", "experiment")
        actions = {
            "experiment": self.train_tune_evaluate,
            "evaluate": self.train_tune_evaluate,
            "build": self.build,
            "search": lambda: self.search(query, method),
            "answer": lambda: self.answer(query),
        }
        if selected not in actions:
            raise ValueError(f"Unknown mode: {selected}. Supported: {list(actions)}")
        return actions[selected]()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LegalRAG training, hyperparameter tuning and evaluation")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--mode", choices=["experiment", "evaluate", "build", "search", "answer"], default=None)
    parser.add_argument("--query", default=None)
    parser.add_argument("--method", choices=list(GraphSearchPipeline.SUPPORTED_METHODS), default="hybrid+graph")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    LegalRAGApplication(args.config).run(args.mode, args.query, args.method)


if __name__ == "__main__":
    main()
