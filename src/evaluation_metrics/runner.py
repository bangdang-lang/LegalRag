from __future__ import annotations

import ast
import itertools
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from .dataset import load_ground_truth
from .metrics import (
    build_metrics_table,
    evaluate_ranked_ids,
    summarize_metrics,
)

_QUOTED_TOKEN_PATTERN = re.compile(r"""['"]([^'"]+)['"]""")

class EvaluationRunner:
    """
    Tune hyperparameter trên 300 train query.

    K không được tune:
        - tuning_k chỉ là K cố định để chọn cấu hình.
        - test cuối chạy tại K = 5, 10, 50, 100, 200.

    Module đồng thời xử lý lỗi ID:
        retrieval chunk_id
            -> graph node_id thông qua source_chunk_ids
            -> graph expansion
            -> source_chunk_ids để so với ground truth.
    """
    
    FIXED_TEST_K_VALUES = [5, 10, 50, 100, 200]

    def __init__(
        self,
        retriever: Any,
        graph_expander: Any,
        config: dict[str, Any],
        project_root: str | Path,
    ) -> None:
        self.retriever = retriever
        self.graph_expander = graph_expander
        self.config = config
        self.project_root = Path(project_root)

        evaluation = config.get("evaluation", {})

        self.tuning_k = int(
            evaluation.get("tuning_k", 10)
        )
        self.primary_metric = str(
            evaluation.get("primary_metric", "mrr@10")
        ).lower()
        self.max_tuning_trials = int(
            evaluation.get("max_tuning_trials", 30)
        )
        self.random_seed = int(
            evaluation.get("random_seed", 42)
        )
        self.log_every = int(
            evaluation.get("log_every", 10)
        )
        self.debug_queries = int(
            evaluation.get("debug_queries", 3)
        )

        output_dir = Path(
            evaluation.get(
                "output_dir",
                "evaluation_output",
            )
        )

        self.output_dir = (
            output_dir
            if output_dir.is_absolute()
            else self.project_root / output_dir
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.chunk_to_node_ids: dict[str, list[str]] = {}
        self._build_chunk_to_node_mapping()

    @staticmethod
    def _as_list(
        value: Any,
        default: list[Any],
    ) -> list[Any]:
        if value is None:
            return default

        return value if isinstance(value, list) else [value]

    def _build_configurations(
        self,
    ) -> list[dict[str, Any]]:
        retrieval = self.config.get("retrieval", {})
        graph = self.config.get("graph", {})

        grid = {
            "bm25_top_k": self._as_list(
                retrieval.get("bm25_top_k"),
                [250],
            ),
            "faiss_top_k": self._as_list(
                retrieval.get("faiss_top_k"),
                [250],
            ),
            "rrf_k": self._as_list(
                retrieval.get("rrf_k"),
                [30, 60, 90],
            ),
            "seed_top_k": self._as_list(
                retrieval.get("seed_top_k"),
                [50, 100, 200, 300],
            ),
            "max_hops": self._as_list(
                graph.get("max_hops"),
                [1, 2],
            ),
            "max_neighbors_per_node": self._as_list(
                graph.get("max_neighbors_per_node"),
                [10, 20, 30],
            ),
            "graph_weight": self._as_list(
                graph.get("graph_weight"),
                [0.1, 0.2, 0.3, 0.4],
            ),
            "hop_decay": self._as_list(
                graph.get("hop_decay"),
                [0.5, 0.65, 0.8],
            ),
        }

        keys = list(grid)

        configurations = [
            dict(zip(keys, values))
            for values in itertools.product(
                *(grid[key] for key in keys)
            )
        ]

        random.Random(
            self.random_seed
        ).shuffle(configurations)

        if self.max_tuning_trials > 0:
            configurations = configurations[
                : self.max_tuning_trials
            ]

        return configurations

    @staticmethod
    def _parse_chunk_ids(value: Any) -> list[str]:
        if value is None:
            return []

        if isinstance(value, (list, tuple, set)):
            return [
                str(item).strip()
                for item in value
                if str(item).strip()
            ]

        if hasattr(value, "tolist"):
            try:
                converted = value.tolist()

                if isinstance(converted, list):
                    return [
                        str(item).strip()
                        for item in converted
                        if str(item).strip()
                    ]
            except Exception:
                pass

        if isinstance(value, str):
            text = value.strip()

            if not text:
                return []

            try:
                parsed = json.loads(text)

                if isinstance(parsed, (list, tuple, set)):
                    return [
                        str(item).strip()
                        for item in parsed
                        if str(item).strip()
                    ]

                return [str(parsed).strip()]
            except json.JSONDecodeError:
                pass

            try:
                parsed = ast.literal_eval(text)

                if isinstance(parsed, (list, tuple, set)):
                    return [
                        str(item).strip()
                        for item in parsed
                        if str(item).strip()
                    ]

                return [str(parsed).strip()]
            except (ValueError, SyntaxError):
                pass

            quoted_tokens = _QUOTED_TOKEN_PATTERN.findall(text)

            if quoted_tokens:
                return [
                    token.strip()
                    for token in quoted_tokens
                    if token.strip()
                ]

            cleaned = text.strip("[](){}")
            tokens = re.split(r"[\s,]+", cleaned)

            return [
                token.strip("'\" ")
                for token in tokens
                if token.strip("'\" ")
            ]

        return [str(value).strip()]

    def _build_chunk_to_node_mapping(self) -> None:
        graph = getattr(
            self.graph_expander,
            "graph",
            None,
        )

        if graph is None:
            raise RuntimeError(
                "GraphExpander chưa load graph. "
                "Hãy gọi GraphExpander.load() trước khi tạo runner."
            )

        mapping: dict[str, list[str]] = defaultdict(list)
        node_count_with_chunks = 0

        for node_id, metadata in graph.nodes(data=True):
            chunk_ids: list[str] = []

            if isinstance(metadata, dict):
                for key in (
                    "source_chunk_ids",
                    "source_chunk_id",
                    "source_chunks",
                ):
                    chunk_ids.extend(
                        self._parse_chunk_ids(
                            metadata.get(key)
                        )
                    )

            if chunk_ids:
                node_count_with_chunks += 1

            for chunk_id in chunk_ids:
                chunk_id = str(chunk_id).strip()
                node_id_text = str(node_id)

                if (
                    chunk_id
                    and node_id_text
                    not in mapping[chunk_id]
                ):
                    mapping[chunk_id].append(
                        node_id_text
                    )

        self.chunk_to_node_ids = dict(mapping)

        print()
        print("=" * 100)
        print("KIỂM TRA ÁNH XẠ CHUNK -> GRAPH NODE")
        print("=" * 100)
        print(
            f"Tổng graph nodes              : "
            f"{graph.number_of_nodes()}"
        )
        print(
            f"Nodes có source_chunk_ids     : "
            f"{node_count_with_chunks}"
        )
        print(
            f"Số chunk ánh xạ được          : "
            f"{len(self.chunk_to_node_ids)}"
        )

        if not self.chunk_to_node_ids:
            print(
                "[CẢNH BÁO] Không tạo được mapping chunk -> node. "
                "Hãy kiểm tra tên cột source_chunk_ids trong nodes.parquet."
            )

    @staticmethod
    def _result_value(
        result: Any,
        key: str,
        default: Any = None,
    ) -> Any:
        if isinstance(result, dict):
            return result.get(key, default)

        return getattr(result, key, default)

    def _map_seed_results_to_graph_nodes(
        self,
        seed_results: list[Any],
    ) -> list[dict[str, Any]]:
        """
        GraphExpander cũ thường yêu cầu seed_id phải là graph node_id.

        Hàm này chuyển chunk_id retrieval sang graph node_id bằng
        source_chunk_ids trong node metadata.
        """
        mapped_results: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()

        for result in seed_results:
            seed_chunk_id = str(
                self._result_value(
                    result,
                    "chunk_id",
                    "",
                )
            )

            if not seed_chunk_id:
                continue

            score = float(
                self._result_value(
                    result,
                    "rrf_score",
                    0.0,
                )
            )

            text = str(
                self._result_value(
                    result,
                    "text",
                    "",
                )
                or ""
            )

            metadata = dict(
                self._result_value(
                    result,
                    "metadata",
                    {},
                )
                or {}
            )

            graph_node_ids: list[str] = []

            graph = self.graph_expander.graph

            if seed_chunk_id in graph:
                graph_node_ids.append(
                    seed_chunk_id
                )

            graph_node_ids.extend(
                self.chunk_to_node_ids.get(
                    seed_chunk_id,
                    [],
                )
            )

            graph_node_ids = list(
                dict.fromkeys(graph_node_ids)
            )

            for graph_node_id in graph_node_ids:
                dedup_key = (
                    seed_chunk_id,
                    graph_node_id,
                )

                if dedup_key in seen:
                    continue

                seen.add(dedup_key)

                node_metadata = dict(
                    graph.nodes[graph_node_id]
                )

                node_metadata.update(metadata)
                node_metadata[
                    "original_seed_chunk_id"
                ] = seed_chunk_id

                mapped_results.append(
                    {
                        "chunk_id": graph_node_id,
                        "rrf_score": score,
                        "text": text,
                        "metadata": node_metadata,
                    }
                )

        return mapped_results

    @classmethod
    def _extract_chunk_ids_from_results(
        cls,
        results: list[Any],
    ) -> list[str]:
        """
        Chuyển graph result về chunk IDs để so với ground truth.

        Không chỉ lấy node_id; ưu tiên source_chunk_ids.
        """
        output: list[str] = []
        seen: set[str] = set()

        for result in results:
            result_id = cls._result_value(
                result,
                "chunk_id",
                None,
            )

            metadata = cls._result_value(
                result,
                "metadata",
                {},
            ) or {}

            candidate_ids: list[str] = []

            if isinstance(metadata, dict):
                for key in (
                    "source_chunk_ids",
                    "source_chunk_id",
                    "source_chunks",
                    "original_seed_chunk_id",
                ):
                    candidate_ids.extend(
                        cls._parse_chunk_ids(
                            metadata.get(key)
                        )
                    )

            if result_id is not None:
                candidate_ids.append(
                    str(result_id)
                )

            for candidate_id in candidate_ids:
                candidate_id = str(
                    candidate_id
                ).strip()

                if (
                    candidate_id
                    and candidate_id not in seen
                ):
                    seen.add(candidate_id)
                    output.append(candidate_id)

        return output

    def _apply_graph_parameters(
        self,
        params: dict[str, Any],
    ) -> None:
        graph_weight = float(
            params["graph_weight"]
        )

        self.graph_expander.graph_weight = (
            graph_weight
        )
        self.graph_expander.retrieval_weight = (
            1.0 - graph_weight
        )
        self.graph_expander.hop_decay = float(
            params["hop_decay"]
        )

    def _search(
        self,
        query: str,
        *,
        final_top_k: int,
        params: dict[str, Any],
    ) -> tuple[list[Any], list[Any], list[Any]]:
        self._apply_graph_parameters(params)

        seed_results = self.retriever.search(
            query=query,
            top_k=int(params["seed_top_k"]),
            bm25_top_k=int(
                params["bm25_top_k"]
            ),
            faiss_top_k=int(
                params["faiss_top_k"]
            ),
            rrf_k=int(params["rrf_k"]),
        )

        mapped_seed_results = (
            self._map_seed_results_to_graph_nodes(
                seed_results
            )
        )

        if mapped_seed_results:
            graph_results = (
                self.graph_expander.expand(
                    mapped_seed_results,
                    final_top_k=final_top_k,
                    max_hops=int(
                        params["max_hops"]
                    ),
                    max_neighbors_per_node=int(
                        params[
                            "max_neighbors_per_node"
                        ]
                    ),
                    include_seeds=True,
                )
            )
        else:
            graph_results = []

        return (
            seed_results,
            mapped_seed_results,
            graph_results,
        )

    def _evaluate_dataset_at_k(
        self,
        dataset: pd.DataFrame,
        *,
        k: int,
        params: dict[str, Any],
        log_prefix: str,
    ) -> tuple[dict[str, float], pd.DataFrame]:
        rows: list[dict[str, Any]] = []
        start_time = time.perf_counter()

        zero_mapping_count = 0
        zero_result_count = 0
        zero_match_count = 0

        for position, (_, sample) in enumerate(
            dataset.iterrows(),
            start=1,
        ):
            query_start = time.perf_counter()

            (
                seed_results,
                mapped_seed_results,
                graph_results,
            ) = self._search(
                str(sample["query"]),
                final_top_k=k,
                params=params,
            )

            if not mapped_seed_results:
                zero_mapping_count += 1

            if not graph_results:
                zero_result_count += 1

            ranked_ids = (
                self._extract_chunk_ids_from_results(
                    graph_results
                )
            )

            relevant_ids = {
                str(item)
                for item in sample[
                    "relevant_chunk_ids"
                ]
            }

            matched_ids = (
                relevant_ids
                & set(ranked_ids)
            )

            if not matched_ids:
                zero_match_count += 1

            metrics = evaluate_ranked_ids(
                ranked_ids,
                relevant_ids,
                k,
            )

            elapsed_query = (
                time.perf_counter()
                - query_start
            )

            rows.append(
                {
                    "query_id": str(
                        sample["query_id"]
                    ),
                    "query": str(
                        sample["query"]
                    ),
                    "k": k,
                    "seed_count": len(
                        seed_results
                    ),
                    "mapped_seed_count": len(
                        mapped_seed_results
                    ),
                    "graph_result_count": len(
                        graph_results
                    ),
                    "returned_chunk_count": len(
                        ranked_ids
                    ),
                    "matched_count": len(
                        matched_ids
                    ),
                    "elapsed_seconds": (
                        elapsed_query
                    ),
                    **metrics,
                }
            )

            if position <= self.debug_queries:
                seed_ids = [
                    str(
                        self._result_value(
                            result,
                            "chunk_id",
                            "",
                        )
                    )
                    for result in seed_results[:10]
                ]

                mapped_ids = [
                    str(
                        self._result_value(
                            result,
                            "chunk_id",
                            "",
                        )
                    )
                    for result in mapped_seed_results[:10]
                ]

                print()
                print(
                    f"{log_prefix} DEBUG QUERY "
                    f"{position}"
                )
                print(
                    f"Query ID             : "
                    f"{sample['query_id']}"
                )
                print(
                    f"Relevant chunk IDs   : "
                    f"{sorted(relevant_ids)}"
                )
                print(
                    f"Retrieval seed IDs   : "
                    f"{seed_ids}"
                )
                print(
                    f"Mapped graph node IDs: "
                    f"{mapped_ids}"
                )
                print(
                    f"Returned chunk IDs   : "
                    f"{ranked_ids[:20]}"
                )
                print(
                    f"Matched IDs          : "
                    f"{sorted(matched_ids)}"
                )
                print(
                    f"Seed/mapped/graph    : "
                    f"{len(seed_results)}/"
                    f"{len(mapped_seed_results)}/"
                    f"{len(graph_results)}"
                )
                print()

            should_log = (
                position == 1
                or position % self.log_every == 0
                or position == len(dataset)
            )

            if should_log:
                elapsed = (
                    time.perf_counter()
                    - start_time
                )
                eta = (
                    elapsed
                    / position
                    * (len(dataset) - position)
                )

                print(
                    f"{log_prefix} "
                    f"{position:>3}/{len(dataset)} | "
                    f"{position / len(dataset) * 100:>6.2f}% | "
                    f"mapped_zero={zero_mapping_count} | "
                    f"result_zero={zero_result_count} | "
                    f"match_zero={zero_match_count} | "
                    f"ETA={eta / 60:>7.2f} phút"
                )

        per_query = pd.DataFrame(rows)

        summary = summarize_metrics(
            per_query,
            k=k,
        )

        summary.update(
            {
                "zero_mapping_queries": (
                    zero_mapping_count
                ),
                "zero_result_queries": (
                    zero_result_count
                ),
                "zero_match_queries": (
                    zero_match_count
                ),
                "elapsed_seconds": (
                    time.perf_counter()
                    - start_time
                ),
            }
        )

        return summary, per_query

    def tune(
        self,
        train_path: str | Path,
    ) -> dict[str, Any]:
        train_dataset = load_ground_truth(
            train_path,
            split="train",
            expected_size=300,
        )

        configurations = (
            self._build_configurations()
        )

        print()
        print("=" * 100)
        print(
            "TUNE HYPERPARAMETER TRÊN "
            "300 TRAIN QUERY"
        )
        print("=" * 100)
        print(
            f"Số cấu hình          : "
            f"{len(configurations)}"
        )
        print(
            f"K cố định khi tune   : "
            f"{self.tuning_k}"
        )
        print(
            f"Metric chọn cấu hình : "
            f"{self.primary_metric}"
        )
        print(
            "K không nằm trong hyperparameter grid."
        )
        print("=" * 100)

        metric_name = (
            self.primary_metric.split("@")[0]
        )

        metric_column_mapping = {
            "hit": "Hit@K",
            "precision": "Precision@K",
            "recall": "Recall@K",
            "mrr": "MRR@K",
            "ndcg": "nDCG@K",
        }

        if metric_name not in metric_column_mapping:
            raise ValueError(
                f"Metric không hỗ trợ: "
                f"{self.primary_metric}"
            )

        metric_column = (
            metric_column_mapping[metric_name]
        )

        best_score = float("-inf")
        best_params: dict[str, Any] | None = None
        tuning_rows: list[dict[str, Any]] = []

        for trial_index, params in enumerate(
            configurations,
            start=1,
        ):
            print()
            print("-" * 100)
            print(
                f"[TRIAL {trial_index}/"
                f"{len(configurations)}]"
            )
            print(
                json.dumps(
                    params,
                    ensure_ascii=False,
                )
            )

            summary, _ = (
                self._evaluate_dataset_at_k(
                    train_dataset,
                    k=self.tuning_k,
                    params=params,
                    log_prefix=(
                        f"[TRIAL {trial_index}]"
                    ),
                )
            )

            score = float(
                summary[metric_column]
            )

            tuning_rows.append(
                {
                    "trial": trial_index,
                    **params,
                    **summary,
                    "selection_score": score,
                }
            )

            print(
                build_metrics_table(
                    [summary]
                ).to_string(
                    index=False,
                    float_format=(
                        lambda value: (
                            f"{value:.6f}"
                        )
                    ),
                )
            )

            if score > best_score:
                best_score = score
                best_params = dict(params)

                print(
                    "=> CẤU HÌNH TỐT NHẤT "
                    "HIỆN TẠI"
                )

        pd.DataFrame(
            tuning_rows
        ).to_csv(
            self.output_dir
            / "train_tuning_results.csv",
            index=False,
            encoding="utf-8-sig",
        )

        if best_params is None:
            raise RuntimeError(
                "Không tìm được cấu hình tốt nhất."
            )

        best_payload = {
            "primary_metric": (
                self.primary_metric
            ),
            "tuning_k": self.tuning_k,
            "best_score": best_score,
            "best_params": best_params,
        }

        (
            self.output_dir
            / "best_hyperparameters.json"
        ).write_text(
            json.dumps(
                best_payload,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print()
        print("=" * 100)
        print("TUNE HOÀN TẤT")
        print("=" * 100)
        print(
            json.dumps(
                best_payload,
                ensure_ascii=False,
                indent=2,
            )
        )

        return best_params

    def evaluate_test(
        self,
        test_path: str | Path,
        *,
        best_params: dict[str, Any],
    ) -> pd.DataFrame:
        test_dataset = load_ground_truth(
            test_path,
            split="test",
            expected_size=100,
        )

        summaries: list[
            dict[str, float]
        ] = []

        print()
        print("#" * 100)
        print(
            "TEST CẤU HÌNH TỐT NHẤT "
            "TRÊN 100 QUERY"
        )
        print("#" * 100)
        print(
            f"K cố định: "
            f"{self.FIXED_TEST_K_VALUES}"
        )
        print(
            json.dumps(
                best_params,
                ensure_ascii=False,
            )
        )
        print("#" * 100)

        for run_index, k in enumerate(
            self.FIXED_TEST_K_VALUES,
            start=1,
        ):
            print()
            print("=" * 100)
            print(
                f"[LƯỢT {run_index}/5] "
                f"TEST K={k}"
            )
            print("=" * 100)

            summary, per_query = (
                self._evaluate_dataset_at_k(
                    test_dataset,
                    k=k,
                    params=best_params,
                    log_prefix=f"[K={k}]",
                )
            )

            summaries.append(summary)

            per_query.to_csv(
                self.output_dir
                / f"test_per_query_k{k}.csv",
                index=False,
                encoding="utf-8-sig",
            )

            print(
                build_metrics_table(
                    [summary]
                ).to_string(
                    index=False,
                    float_format=(
                        lambda value: (
                            f"{value:.6f}"
                        )
                    ),
                )
            )

        final_table = build_metrics_table(
            summaries
        )

        final_table.to_csv(
            self.output_dir
            / "test_metrics_by_k.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print()
        print("=" * 100)
        print("BẢNG METRICS CUỐI")
        print("=" * 100)
        print(
            final_table.to_string(
                index=False,
                float_format=(
                    lambda value: (
                        f"{value:.6f}"
                    )
                ),
            )
        )

        return final_table
