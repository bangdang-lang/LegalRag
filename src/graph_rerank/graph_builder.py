from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

import pandas as pd

from core.config import AppConfig


class LegalHierarchyGraphBuilder:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.settings = config.section("graph")
        self.logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def _node_id(prefix: str, value: str) -> str:
        digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:20]
        return f"{prefix}:{digest}"

    @staticmethod
    def _chunk_type(chunk_id: str) -> str:
        value = chunk_id.lower()
        if "_dieu_" in value:
            return "article"
        if "_phu_luc_" in value:
            return "appendix"
        if "_bang_" in value:
            return "table"
        return "chunk"

    def build(self) -> tuple[Path, Path]:
        chunks = pd.read_parquet(self.config.path("legal_chunks"))
        required = {"id", "document_id", "content"}
        missing = required - set(chunks.columns)
        if missing:
            raise ValueError(f"Missing graph columns: {sorted(missing)}")
        nodes: dict[str, dict] = {}
        edges: dict[tuple[str, str, str], dict] = {}
        for _, row in chunks.iterrows():
            chunk_id = str(row["id"]).strip()
            document_id = str(row["document_id"]).strip()
            document_node = self._node_id("document", document_id)
            nodes.setdefault(document_node, self._make_node(document_node, "document", row, [], None))
            chunk_type = self._chunk_type(chunk_id)
            if chunk_type != "article" and not self.settings.get("include_non_article_chunks", True):
                continue
            child_node = self._node_id(chunk_type, chunk_id)
            nodes[child_node] = self._make_node(child_node, chunk_type, row, [chunk_id], document_node)
            edge_type = {"appendix": "contains_appendix", "table": "contains_table"}.get(chunk_type, "contains")
            weight = float(self.settings.get("edge_weights", {}).get(edge_type, 1.0))
            edges[(document_node, child_node, edge_type)] = {
                self.settings.get("source_column", "source_node_id"): document_node,
                self.settings.get("target_column", "target_node_id"): child_node,
                "edge_type": edge_type,
                "reference_count": weight,
            }
        node_frame = pd.DataFrame(nodes.values())
        edge_frame = pd.DataFrame(edges.values())
        node_path, edge_path = self.config.path("graph_nodes"), self.config.path("graph_edges")
        node_path.parent.mkdir(parents=True, exist_ok=True)
        node_frame.to_parquet(node_path, index=False)
        edge_frame.to_parquet(edge_path, index=False)
        self.logger.info("Graph built: %d nodes, %d edges, %d mapped chunks", len(node_frame), len(edge_frame), sum(node_frame["source_chunk_ids"].ne("[]")))
        return node_path, edge_path

    def _make_node(self, node_id: str, node_type: str, row: pd.Series, chunk_ids: list[str], parent: str | None) -> dict:
        copied = {key: row.get(key) for key in ["document_id", "document_number", "url", "legal_type", "legal_sectors", "issuing_authority", "issuance_date", "signers"]}
        return {
            "node_id": node_id,
            "node_type": node_type,
            "label": str(row.get("title") or row.get("id") or node_id),
            "parent_node_id": parent,
            "is_virtual": node_type == "document",
            "article_number": self._extract_article_number(str(row.get("id", ""))),
            "content": "" if node_type == "document" else str(row.get("content", "")),
            "source_chunk_ids": json.dumps(chunk_ids, ensure_ascii=False),
            "resolution_status": "resolved",
            **copied,
        }

    @staticmethod
    def _extract_article_number(chunk_id: str) -> str | None:
        match = re.search(r"_dieu_([^_]+)", chunk_id.lower())
        return match.group(1) if match else None
