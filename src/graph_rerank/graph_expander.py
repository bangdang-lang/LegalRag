from __future__ import annotations

import ast
import json
import pickle
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

from .schemas import GraphSearchResult

_QUOTED_TOKEN_PATTERN = re.compile(r"""['"]([^'"]+)['"]""")

class GraphExpander:
    """
    Mở rộng kết quả retrieval trên graph và xếp hạng lại.

    Hỗ trợ trường hợp:
        - Retrieval trả về chunk_id.
        - Graph dùng node_id riêng.
        - Node graph lưu các chunk nguồn trong source_chunk_ids.

    Quy trình:
        chunk_id retrieval
            -> ánh xạ sang graph node_id
            -> mở rộng graph
            -> trả kết quả kèm source_chunk_ids
    """

    TEXT_KEYS = (
        "text",
        "chunk_text",
        "content",
        "chunk",
        "page_content",
    )

    SOURCE_CHUNK_KEYS = (
        "source_chunk_ids",
        "source_chunk_id",
        "source_chunks",
    )

    def __init__(
        self,
        graph_path: str | Path,
        *,
        retrieval_weight: float = 0.70,
        graph_weight: float = 0.30,
        hop_decay: float = 0.65,
        direction: str = "both",
        edge_weight_key: str = "weight",
        edge_type_key: str = "type",
        allowed_edge_types: set[str] | None = None,
    ) -> None:
        self.graph_path = Path(graph_path)

        total = retrieval_weight + graph_weight

        if total <= 0:
            raise ValueError(
                "Tổng retrieval_weight và graph_weight phải > 0."
            )

        if not 0 < hop_decay <= 1:
            raise ValueError(
                "hop_decay phải nằm trong khoảng (0, 1]."
            )

        if direction not in {"out", "in", "both"}:
            raise ValueError(
                "direction phải là 'out', 'in' hoặc 'both'."
            )

        self.retrieval_weight = retrieval_weight / total
        self.graph_weight = graph_weight / total
        self.hop_decay = hop_decay
        self.direction = direction
        self.edge_weight_key = edge_weight_key
        self.edge_type_key = edge_type_key
        self.allowed_edge_types = allowed_edge_types

        self.graph: nx.Graph | nx.DiGraph | None = None

        # chunk_id -> danh sách graph node_id chứa chunk đó.
        self.chunk_to_node_ids: dict[str, list[str]] = {}

    def load(self) -> "GraphExpander":
        """
        Đọc graph từ:

        1. Thư mục chứa:
           - nodes.parquet
           - edges.parquet

        2. File:
           - .graphml
           - .gexf
           - .json
           - .pkl
           - .pickle
           - .gpickle
        """
        if not self.graph_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy graph: {self.graph_path.resolve()}"
            )

        if self.graph_path.is_dir():
            graph = self._load_from_parquet_directory(
                self.graph_path
            )
        else:
            graph = self._load_from_file(
                self.graph_path
            )

        self.graph = nx.relabel_nodes(
            graph,
            {
                node: str(node)
                for node in graph.nodes
            },
            copy=True,
        )

        self._build_chunk_to_node_mapping()

        print()
        print("=" * 100)
        print("GRAPH LOADED")
        print("=" * 100)
        print(
            f"Nodes                       : "
            f"{self.graph.number_of_nodes()}"
        )
        print(
            f"Edges                       : "
            f"{self.graph.number_of_edges()}"
        )
        print(
            f"Chunk -> node mappings      : "
            f"{len(self.chunk_to_node_ids)}"
        )

        if not self.chunk_to_node_ids:
            print(
                "[CẢNH BÁO] Không tìm thấy source_chunk_ids trong "
                "metadata node. Graph chỉ mở rộng được nếu seed_id "
                "trùng trực tiếp node_id."
            )

        return self

    def _load_from_file(
        self,
        graph_path: Path,
    ) -> nx.Graph | nx.DiGraph:
        suffix = graph_path.suffix.lower()

        if suffix == ".graphml":
            return nx.read_graphml(graph_path)

        if suffix == ".gexf":
            return nx.read_gexf(graph_path)

        if suffix in {
            ".pkl",
            ".pickle",
            ".gpickle",
        }:
            with graph_path.open("rb") as file:
                return pickle.load(file)

        if suffix == ".json":
            with graph_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                payload = json.load(file)

            return nx.node_link_graph(payload)

        raise ValueError(
            "Graph phải là thư mục chứa nodes.parquet và "
            "edges.parquet, hoặc file .graphml, .gexf, .json, "
            ".pkl, .pickle, .gpickle."
        )

    def _load_from_parquet_directory(
        self,
        graph_directory: Path,
    ) -> nx.DiGraph:
        nodes_path = graph_directory / "nodes.parquet"
        edges_path = graph_directory / "edges.parquet"

        if not nodes_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy file nodes.parquet:\n{nodes_path}"
            )

        if not edges_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy file edges.parquet:\n{edges_path}"
            )

        nodes_df = pd.read_parquet(nodes_path)
        edges_df = pd.read_parquet(edges_path)

        print(
            "Node columns:",
            list(nodes_df.columns),
        )
        print(
            "Edge columns:",
            list(edges_df.columns),
        )

        node_id_column = self._find_first_column(
            nodes_df,
            [
                "node_id",
                "id",
                "chunk_id",
                "graph_node_id",
            ],
        )

        source_column = self._find_first_column(
            edges_df,
            [
                "source",
                "source_id",
                "src",
                "from",
                "from_id",
            ],
        )

        target_column = self._find_first_column(
            edges_df,
            [
                "target",
                "target_id",
                "dst",
                "to",
                "to_id",
            ],
        )

        graph = nx.DiGraph()

        for _, row in nodes_df.iterrows():
            node_id = str(
                row[node_id_column]
            )

            attributes: dict[str, Any] = {}

            for column, value in row.items():
                if column == node_id_column:
                    continue

                if self._is_missing(value):
                    continue

                attributes[str(column)] = (
                    self._convert_value(value)
                )

            graph.add_node(
                node_id,
                **attributes,
            )

        for _, row in edges_df.iterrows():
            source_id = str(
                row[source_column]
            )
            target_id = str(
                row[target_column]
            )

            attributes: dict[str, Any] = {}

            for column, value in row.items():
                if column in {
                    source_column,
                    target_column,
                }:
                    continue

                if self._is_missing(value):
                    continue

                attributes[str(column)] = (
                    self._convert_value(value)
                )

            graph.add_edge(
                source_id,
                target_id,
                **attributes,
            )

        return graph

    @staticmethod
    def _is_missing(
        value: Any,
    ) -> bool:
        """
        Tránh lỗi pd.isna(list) trả về mảng boolean.
        """
        if value is None:
            return True

        if isinstance(
            value,
            (
                list,
                tuple,
                set,
                dict,
            ),
        ):
            return False

        try:
            result = pd.isna(value)

            if isinstance(result, bool):
                return result

            return False
        except Exception:
            return False

    @staticmethod
    def _find_first_column(
        dataframe: pd.DataFrame,
        candidates: list[str],
    ) -> str:
        columns_lower = {
            str(column).lower(): str(column)
            for column in dataframe.columns
        }

        for candidate in candidates:
            if candidate.lower() in columns_lower:
                return columns_lower[
                    candidate.lower()
                ]

        raise ValueError(
            f"Không tìm thấy cột phù hợp.\n"
            f"Các cột hiện có: "
            f"{list(dataframe.columns)}\n"
            f"Cần một trong các cột: "
            f"{candidates}"
        )

    @staticmethod
    def _convert_value(
        value: Any,
    ) -> Any:
        if hasattr(value, "tolist"):
            try:
                return value.tolist()
            except Exception:
                pass

        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass

        return value

    @staticmethod
    def _parse_chunk_ids(value: Any) -> list[str]:
        """
        Hỗ trợ cả chuỗi NumPy array không có dấu phẩy:
        "['chunk_1' 'chunk_2']"
        """
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

    def _node_source_chunk_ids(
        self,
        node_id: str,
    ) -> list[str]:
        if self.graph is None:
            return []

        if node_id not in self.graph:
            return []

        metadata = dict(
            self.graph.nodes[node_id]
        )

        output: list[str] = []
        seen: set[str] = set()

        for key in self.SOURCE_CHUNK_KEYS:
            for chunk_id in self._parse_chunk_ids(
                metadata.get(key)
            ):
                chunk_id = str(
                    chunk_id
                ).strip()

                if (
                    chunk_id
                    and chunk_id not in seen
                ):
                    seen.add(chunk_id)
                    output.append(chunk_id)

        return output

    def _build_chunk_to_node_mapping(
        self,
    ) -> None:
        if self.graph is None:
            raise RuntimeError(
                "Graph chưa được load."
            )

        mapping: dict[
            str,
            list[str],
        ] = defaultdict(list)

        nodes_with_source_chunks = 0

        for node_id, metadata in (
            self.graph.nodes(data=True)
        ):
            source_chunk_ids: list[str] = []

            if isinstance(metadata, dict):
                for key in self.SOURCE_CHUNK_KEYS:
                    source_chunk_ids.extend(
                        self._parse_chunk_ids(
                            metadata.get(key)
                        )
                    )

            source_chunk_ids = list(
                dict.fromkeys(
                    str(chunk_id).strip()
                    for chunk_id
                    in source_chunk_ids
                    if str(chunk_id).strip()
                )
            )

            if source_chunk_ids:
                nodes_with_source_chunks += 1

            for chunk_id in source_chunk_ids:
                node_id_text = str(node_id)

                if (
                    node_id_text
                    not in mapping[chunk_id]
                ):
                    mapping[chunk_id].append(
                        node_id_text
                    )

        self.chunk_to_node_ids = dict(
            mapping
        )

        print(
            f"Nodes có source_chunk_ids : "
            f"{nodes_with_source_chunks}"
        )

    @staticmethod
    def _normalize(
        scores: dict[str, float],
    ) -> dict[str, float]:
        if not scores:
            return {}

        minimum = min(scores.values())
        maximum = max(scores.values())

        if maximum == minimum:
            return {
                key: 1.0
                for key in scores
            }

        return {
            key: (
                value - minimum
            )
            / (
                maximum - minimum
            )
            for key, value in scores.items()
        }

    def _edge_allowed(
        self,
        data: dict[str, Any],
    ) -> bool:
        if self.allowed_edge_types is None:
            return True

        return (
            data.get(
                self.edge_type_key
            )
            in self.allowed_edge_types
        )

    def _edge_weight(
        self,
        data: dict[str, Any],
    ) -> float:
        try:
            return max(
                float(
                    data.get(
                        self.edge_weight_key,
                        1.0,
                    )
                ),
                0.0,
            )
        except (
            TypeError,
            ValueError,
        ):
            return 1.0

    def _neighbors(
        self,
        node_id: str,
    ) -> list[
        tuple[
            str,
            dict[str, Any],
        ]
    ]:
        if self.graph is None:
            raise RuntimeError(
                "Graph chưa được load."
            )

        node_id = str(node_id)

        if node_id not in self.graph:
            return []

        output: list[
            tuple[
                str,
                dict[str, Any],
            ]
        ] = []

        if not self.graph.is_directed():
            for neighbor in (
                self.graph.neighbors(node_id)
            ):
                data = (
                    self.graph.get_edge_data(
                        node_id,
                        neighbor,
                    )
                    or {}
                )

                output.append(
                    (
                        str(neighbor),
                        dict(data),
                    )
                )

            return output

        if self.direction in {
            "out",
            "both",
        }:
            for neighbor in (
                self.graph.successors(
                    node_id
                )
            ):
                data = (
                    self.graph.get_edge_data(
                        node_id,
                        neighbor,
                    )
                    or {}
                )

                output.append(
                    (
                        str(neighbor),
                        dict(data),
                    )
                )

        if self.direction in {
            "in",
            "both",
        }:
            for neighbor in (
                self.graph.predecessors(
                    node_id
                )
            ):
                data = (
                    self.graph.get_edge_data(
                        neighbor,
                        node_id,
                    )
                    or {}
                )

                output.append(
                    (
                        str(neighbor),
                        dict(data),
                    )
                )

        return output

    @staticmethod
    def _result_value(
        result: Any,
        name: str,
        default: Any = None,
    ) -> Any:
        if isinstance(result, dict):
            return result.get(
                name,
                default,
            )

        return getattr(
            result,
            name,
            default,
        )

    def _resolve_seed_nodes(
        self,
        seed_chunk_id: str,
    ) -> list[str]:
        """
        Trả về graph node IDs có thể dùng làm điểm bắt đầu.

        Ưu tiên:
        1. seed_chunk_id trùng trực tiếp node_id.
        2. Mapping từ source_chunk_ids.
        """
        if self.graph is None:
            return []

        start_node_ids: list[str] = []

        if seed_chunk_id in self.graph:
            start_node_ids.append(
                seed_chunk_id
            )

        start_node_ids.extend(
            self.chunk_to_node_ids.get(
                seed_chunk_id,
                [],
            )
        )

        return list(
            dict.fromkeys(
                start_node_ids
            )
        )

    def expand(
        self,
        seed_results: list[Any],
        *,
        final_top_k: int = 10,
        max_hops: int = 1,
        max_neighbors_per_node: int = 20,
        include_seeds: bool = True,
    ) -> list[GraphSearchResult]:
        """
        seed_results có thể chứa chunk_id retrieval hoặc node_id graph.

        Nếu seed là chunk_id, module tự ánh xạ sang node_id bằng
        source_chunk_ids trong metadata graph.
        """
        if self.graph is None:
            raise RuntimeError(
                "Hãy gọi GraphExpander.load() trước."
            )

        if not seed_results:
            return []

        if final_top_k <= 0:
            return []

        raw_node_retrieval_scores: dict[
            str,
            float,
        ] = {}

        node_texts: dict[
            str,
            str,
        ] = {}

        node_metadata: dict[
            str,
            dict[str, Any],
        ] = {}

        node_original_seed_chunks: dict[
            str,
            set[str],
        ] = defaultdict(set)

        unmapped_seed_count = 0

        for result in seed_results:
            seed_chunk_id = str(
                self._result_value(
                    result,
                    "chunk_id",
                    "",
                )
            ).strip()

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

            start_node_ids = (
                self._resolve_seed_nodes(
                    seed_chunk_id
                )
            )

            if not start_node_ids:
                unmapped_seed_count += 1
                continue

            for node_id in start_node_ids:
                previous_score = (
                    raw_node_retrieval_scores.get(
                        node_id,
                        0.0,
                    )
                )

                raw_node_retrieval_scores[
                    node_id
                ] = max(
                    previous_score,
                    score,
                )

                if (
                    text
                    and node_id
                    not in node_texts
                ):
                    node_texts[node_id] = text

                combined_metadata = dict(
                    self.graph.nodes[node_id]
                )
                combined_metadata.update(
                    metadata
                )

                source_chunk_ids = (
                    self._node_source_chunk_ids(
                        node_id
                    )
                )

                if seed_chunk_id not in (
                    source_chunk_ids
                ):
                    source_chunk_ids.append(
                        seed_chunk_id
                    )

                combined_metadata[
                    "source_chunk_ids"
                ] = list(
                    dict.fromkeys(
                        source_chunk_ids
                    )
                )

                combined_metadata[
                    "original_seed_chunk_id"
                ] = seed_chunk_id

                node_metadata[node_id] = (
                    combined_metadata
                )

                node_original_seed_chunks[
                    node_id
                ].add(seed_chunk_id)

        if not raw_node_retrieval_scores:
            print(
                "[CẢNH BÁO] Không có retrieval seed nào ánh xạ "
                "được sang graph node."
            )
            print(
                f"Unmapped seeds: "
                f"{unmapped_seed_count}/"
                f"{len(seed_results)}"
            )
            return []

        normalized_retrieval = (
            self._normalize(
                raw_node_retrieval_scores
            )
        )

        graph_scores: dict[
            str,
            float,
        ] = defaultdict(float)

        source_seed_nodes: dict[
            str,
            set[str],
        ] = defaultdict(set)

        source_seed_chunks: dict[
            str,
            set[str],
        ] = defaultdict(set)

        minimum_hops: dict[
            str,
            int,
        ] = {}

        for (
            seed_node_id,
            seed_score,
        ) in normalized_retrieval.items():
            original_chunk_ids = (
                node_original_seed_chunks.get(
                    seed_node_id,
                    set(),
                )
            )

            if include_seeds:
                graph_scores[
                    seed_node_id
                ] += seed_score

                source_seed_nodes[
                    seed_node_id
                ].add(seed_node_id)

                source_seed_chunks[
                    seed_node_id
                ].update(
                    original_chunk_ids
                )

                minimum_hops[
                    seed_node_id
                ] = 0

            queue = deque(
                [
                    (
                        seed_node_id,
                        0,
                        1.0,
                    )
                ]
            )

            best_hop = {
                seed_node_id: 0
            }

            while queue:
                (
                    current_id,
                    current_hop,
                    path_weight,
                ) = queue.popleft()

                if current_hop >= max_hops:
                    continue

                neighbors = [
                    pair
                    for pair in self._neighbors(
                        current_id
                    )
                    if self._edge_allowed(
                        pair[1]
                    )
                ]

                neighbors.sort(
                    key=lambda pair: (
                        self._edge_weight(
                            pair[1]
                        )
                    ),
                    reverse=True,
                )

                neighbors = neighbors[
                    :max_neighbors_per_node
                ]

                next_hop = (
                    current_hop + 1
                )

                for (
                    neighbor_id,
                    edge_data,
                ) in neighbors:
                    edge_weight = (
                        self._edge_weight(
                            edge_data
                        )
                    )

                    next_path_weight = (
                        path_weight
                        * edge_weight
                    )

                    contribution = (
                        seed_score
                        * next_path_weight
                        * (
                            self.hop_decay
                            ** next_hop
                        )
                    )

                    graph_scores[
                        neighbor_id
                    ] += contribution

                    source_seed_nodes[
                        neighbor_id
                    ].add(seed_node_id)

                    source_seed_chunks[
                        neighbor_id
                    ].update(
                        original_chunk_ids
                    )

                    old_hop = (
                        minimum_hops.get(
                            neighbor_id
                        )
                    )

                    if (
                        old_hop is None
                        or next_hop
                        < old_hop
                    ):
                        minimum_hops[
                            neighbor_id
                        ] = next_hop

                    previously_seen = (
                        best_hop.get(
                            neighbor_id
                        )
                    )

                    if (
                        previously_seen
                        is None
                        or next_hop
                        < previously_seen
                    ):
                        best_hop[
                            neighbor_id
                        ] = next_hop

                        queue.append(
                            (
                                neighbor_id,
                                next_hop,
                                next_path_weight,
                            )
                        )

        normalized_graph = self._normalize(
            dict(graph_scores)
        )

        candidate_node_ids = set(
            graph_scores
        )

        if include_seeds:
            candidate_node_ids.update(
                raw_node_retrieval_scores
            )

        output: list[
            GraphSearchResult
        ] = []

        for node_id in candidate_node_ids:
            retrieval_score = (
                normalized_retrieval.get(
                    node_id,
                    0.0,
                )
            )

            graph_score = (
                normalized_graph.get(
                    node_id,
                    0.0,
                )
            )

            final_score = (
                self.retrieval_weight
                * retrieval_score
                + self.graph_weight
                * graph_score
            )

            metadata = dict(
                self.graph.nodes[node_id]
            )

            metadata.update(
                node_metadata.get(
                    node_id,
                    {},
                )
            )

            node_source_chunks = (
                self._node_source_chunk_ids(
                    node_id
                )
            )

            node_source_chunks.extend(
                source_seed_chunks.get(
                    node_id,
                    set(),
                )
            )

            metadata[
                "source_chunk_ids"
            ] = list(
                dict.fromkeys(
                    str(chunk_id)
                    for chunk_id
                    in node_source_chunks
                    if str(chunk_id).strip()
                )
            )

            metadata[
                "graph_node_id"
            ] = node_id

            text = node_texts.get(
                node_id,
                "",
            )

            if not text:
                for key in self.TEXT_KEYS:
                    value = metadata.get(key)

                    if value:
                        text = str(value)
                        break

            output.append(
                GraphSearchResult(
                    chunk_id=node_id,
                    final_score=final_score,
                    retrieval_score=(
                        retrieval_score
                    ),
                    graph_score=graph_score,
                    hop=minimum_hops.get(
                        node_id
                    ),
                    source_seed_ids=sorted(
                        source_seed_chunks.get(
                            node_id,
                            set(),
                        )
                    ),
                    text=text,
                    metadata=metadata,
                )
            )

        output.sort(
            key=lambda item: (
                -item.final_score,
                -item.retrieval_score,
                -item.graph_score,
                (
                    item.hop
                    if item.hop
                    is not None
                    else 10**9
                ),
                item.chunk_id,
            )
        )

        return output[:final_top_k]
