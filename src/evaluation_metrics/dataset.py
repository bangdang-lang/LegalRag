from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


LIST_COLUMNS = (
    "relevant_chunk_ids",
    "relevant_document_ids",
    "relevant_article_ids",
)


_QUOTED_TOKEN_PATTERN = re.compile(r"""['"]([^'"]+)['"]""")


def parse_id_list(value: Any) -> list[str]:
    """
    Chuẩn hóa danh sách ID từ:
    - list / tuple / set
    - numpy.ndarray
    - JSON string
    - Python list string
    - NumPy array string không có dấu phẩy:
      "['id_1' 'id_2']"
    """
    if value is None:
        return []

    if isinstance(value, float) and pd.isna(value):
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

        # Trường hợp JSON chuẩn.
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

        # Trường hợp Python list chuẩn.
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

        # Quan trọng: chuỗi ndarray kiểu:
        # "['51881_xxx' '51881_yyy']"
        quoted_tokens = _QUOTED_TOKEN_PATTERN.findall(text)

        if quoted_tokens:
            return [
                token.strip()
                for token in quoted_tokens
                if token.strip()
            ]

        # Fallback cho chuỗi phân tách bằng dấu cách/dấu phẩy.
        cleaned = text.strip("[](){}")
        tokens = re.split(r"[\s,]+", cleaned)

        return [
            token.strip("'\" ")
            for token in tokens
            if token.strip("'\" ")
        ]

    return [str(value).strip()]


def load_ground_truth(
    path: str | Path,
    *,
    split: str | None = None,
    expected_size: int | None = None,
) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy ground truth: {path.resolve()}"
        )

    suffix = path.suffix.lower()

    if suffix == ".parquet":
        dataframe = pd.read_parquet(path)
    elif suffix == ".jsonl":
        dataframe = pd.read_json(path, lines=True)
    else:
        raise ValueError(
            "Ground truth phải là file .parquet hoặc .jsonl."
        )

    for column in LIST_COLUMNS:
        if column in dataframe.columns:
            dataframe[column] = dataframe[column].apply(parse_id_list)

    if split is not None and "split" in dataframe.columns:
        dataframe = dataframe[
            dataframe["split"].astype(str).str.lower()
            == split.lower()
        ].reset_index(drop=True)

    required_columns = {
        "query_id",
        "query",
        "relevant_chunk_ids",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Ground truth thiếu các cột: {sorted(missing_columns)}"
        )

    dataframe["query_id"] = dataframe["query_id"].astype(str)
    dataframe["query"] = dataframe["query"].astype(str)

    dataframe = dataframe[
        dataframe["query"].str.strip().ne("")
        & dataframe["relevant_chunk_ids"].map(bool)
    ].reset_index(drop=True)

    if expected_size is not None and len(dataframe) != expected_size:
        raise ValueError(
            f"Số query sau khi lọc là {len(dataframe)}, "
            f"mong đợi {expected_size}."
        )

    return dataframe
