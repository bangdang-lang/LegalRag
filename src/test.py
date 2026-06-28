import pandas as pd

bm25 = pd.read_parquet("bm25_output/bm25_lookup.parquet")
emb = pd.read_parquet("embedding_output/embedding_metadata.parquet")

print(len(bm25), len(emb))
print("Same ID order:", bm25["id"].astype(str).equals(emb["id"].astype(str)))

mismatch = (
    bm25["id"].astype(str).reset_index(drop=True)
    != emb["id"].astype(str).reset_index(drop=True)
)

print("Mismatched rows:", mismatch.sum())

nodes = pd.read_parquet("graph/nodes.parquet")

print(nodes.columns.tolist())
print(nodes[["node_id", "source_chunk_ids"]].head())
print("Missing source_chunk_ids:", nodes["source_chunk_ids"].isna().sum())

import pandas as pd

nodes = pd.read_parquet("graph/nodes.parquet")

value = nodes.loc[0, "source_chunk_ids"]

print("Value:", value)
print("Type:", type(value))
print("repr:", repr(value))

import ast
import json
import pandas as pd


def parse_chunk_ids(value):
    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        return [str(x).strip() for x in value if str(x).strip()]

    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass

    text = str(value).strip()

    if not text:
        return []

    # Ưu tiên parse JSON
    try:
        parsed = json.loads(text)

        if isinstance(parsed, list):
            return [
                str(x).strip()
                for x in parsed
                if str(x).strip()
            ]

        if parsed is not None:
            return [str(parsed).strip()]

    except (json.JSONDecodeError, TypeError):
        pass

    # Hỗ trợ chuỗi dạng Python list
    try:
        parsed = ast.literal_eval(text)

        if isinstance(parsed, (list, tuple, set)):
            return [
                str(x).strip()
                for x in parsed
                if str(x).strip()
            ]

        if parsed is not None:
            return [str(parsed).strip()]

    except (ValueError, SyntaxError):
        pass

    # Fallback cho chuỗi phân cách bằng dấu phẩy
    return [
        item.strip().strip("'\"")
        for item in text.strip("[]").split(",")
        if item.strip().strip("'\"")
    ]


nodes = pd.read_parquet("graph/nodes.parquet")
retrieval_metadata = pd.read_parquet(
    "embedding_output/embedding_metadata.parquet"
)

retrieval_ids = set(
    retrieval_metadata["id"]
    .astype(str)
    .str.strip()
)

graph_chunk_ids = set()

for value in nodes["source_chunk_ids"]:
    graph_chunk_ids.update(parse_chunk_ids(value))

matched_ids = retrieval_ids & graph_chunk_ids
missing_in_graph = retrieval_ids - graph_chunk_ids
unknown_in_graph = graph_chunk_ids - retrieval_ids

print("Retrieval IDs       :", len(retrieval_ids))
print("Graph source IDs    :", len(graph_chunk_ids))
print("Matched IDs         :", len(matched_ids))
print("Missing in graph    :", len(missing_in_graph))
print("Unknown graph IDs   :", len(unknown_in_graph))
print(
    "Mapping coverage    :",
    len(matched_ids) / max(len(retrieval_ids), 1)
)

print("\nVí dụ ID retrieval không có trong graph:")
print(list(sorted(missing_in_graph))[:20])

print("\nVí dụ ID graph không tồn tại trong retrieval:")
print(list(sorted(unknown_in_graph))[:20])

import re
import pandas as pd


def get_chunk_type(chunk_id: str) -> str:
    chunk_id = str(chunk_id).lower()

    patterns = [
        "dieu",
        "khoan",
        "diem",
        "phu_luc",
        "bang",
        "chuong",
        "muc",
        "phan",
        "noi_nhan",
        "chu_ky",
    ]

    for pattern in patterns:
        if f"_{pattern}_" in chunk_id:
            return pattern

    return "other"


retrieval_metadata = pd.read_parquet(
    "embedding_output/embedding_metadata.parquet"
)

retrieval_metadata["id"] = (
    retrieval_metadata["id"]
    .astype(str)
    .str.strip()
)

retrieval_metadata["chunk_type"] = (
    retrieval_metadata["id"]
    .map(get_chunk_type)
)

retrieval_metadata["in_graph"] = (
    retrieval_metadata["id"]
    .isin(graph_chunk_ids)
)

coverage_by_type = (
    retrieval_metadata
    .groupby("chunk_type")
    .agg(
        total=("id", "size"),
        mapped=("in_graph", "sum"),
    )
    .reset_index()
)

coverage_by_type["missing"] = (
    coverage_by_type["total"]
    - coverage_by_type["mapped"]
)

coverage_by_type["coverage"] = (
    coverage_by_type["mapped"]
    / coverage_by_type["total"]
)

coverage_by_type = coverage_by_type.sort_values(
    "total",
    ascending=False,
)

print(coverage_by_type.to_string(index=False))

edges = pd.read_parquet("graph/edges.parquet")

print("Các cột trong edges.parquet:")
print(edges.columns.tolist())