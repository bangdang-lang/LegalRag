"""
build_bm25_index_manual.py

Xây BM25 index cho legal_chunks.parquet bằng bm25s.

Đặc điểm:
- Không dùng argparse.
- Không dùng hàm main().
- Gọi build_bm25_index(...) trực tiếp ở cuối file.
- Không cần GPU.
- Giữ nguyên thứ tự dòng:
      bm25_row == FAISS id == dòng trong legal_chunks.parquet
- Lưu BM25 index, vocabulary, stopwords, bảng tra cứu và manifest.

Cài đặt:
    pip install -U "bm25s[core]>=0.2.0" pyarrow numpy

Chạy:
    python build_bm25_index_manual.py
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import time
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


LOGGER = logging.getLogger("legal_bm25_builder")

TEXT_COLUMNS = [
    "document_number",
    "title",
    "part",
    "chapter",
    "section",
    "articles",
    "content",
]

LOOKUP_COLUMNS = [
    "id",
    "document_id",
    "document_number",
    "title",
    "legal_type",
    "legal_sectors",
    "issuing_authority",
    "issuance_date",
    "part",
    "chapter",
    "section",
    "articles",
    "content",
]


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


def require_bm25s() -> tuple[Any, Any]:
    try:
        import bm25s
        from bm25s.tokenization import Tokenizer
    except ImportError as error:
        raise RuntimeError(
            'Thiếu bm25s. Cài bằng:\n'
            'pip install -U "bm25s[core]>=0.2.0" pyarrow numpy'
        ) from error

    return bm25s, Tokenizer


def save_json(path: Path, data: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    temporary_path.replace(path)


def safe_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return ""

    text = unicodedata.normalize("NFKC", str(value))
    text = re.sub(r"\s+", " ", text).strip()

    if text.lower() in {"", "none", "null", "nan"}:
        return ""

    return text


def build_contextual_text(record: dict[str, Any]) -> str:
    """
    Ghép metadata cấu trúc với nội dung để BM25 nhận biết:
    số hiệu, tên văn bản, phần, chương, mục và điều.
    """
    hierarchy = [
        ("Số hiệu văn bản", record.get("document_number")),
        ("Tên văn bản", record.get("title")),
        ("Phần", record.get("part")),
        ("Chương", record.get("chapter")),
        ("Mục", record.get("section")),
        ("Điều", record.get("articles")),
    ]

    parts: list[str] = []

    for label, value in hierarchy:
        normalized = safe_text(value)
        if normalized:
            parts.append(f"{label}: {normalized}")

    content = safe_text(record.get("content"))

    if content:
        parts.append(f"Nội dung: {content}")

    return "\n".join(parts) or "[EMPTY LEGAL CHUNK]"


def batch_to_texts(
    batch: pa.RecordBatch,
    text_mode: str,
) -> list[str]:
    columns = batch.to_pydict()
    texts: list[str] = []

    for row_index in range(batch.num_rows):
        if text_mode == "content":
            content = safe_text(columns["content"][row_index])
            texts.append(content or "[EMPTY LEGAL CHUNK]")
            continue

        record = {
            column: values[row_index]
            for column, values in columns.items()
        }
        texts.append(build_contextual_text(record))

    return texts


def validate_input(
    input_path: Path,
    text_mode: str,
) -> tuple[list[str], int]:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file: {input_path.resolve()}"
        )

    parquet_file = pq.ParquetFile(input_path)
    available_columns = parquet_file.schema_arrow.names
    row_count = parquet_file.metadata.num_rows

    required_columns = ["id", "content"]
    missing_columns = [
        column
        for column in required_columns
        if column not in available_columns
    ]

    if missing_columns:
        raise ValueError(
            f"File thiếu cột bắt buộc: {missing_columns}\n"
            f"Các cột hiện có: {available_columns}"
        )

    if row_count <= 0:
        raise ValueError("File parquet không có dòng dữ liệu.")

    if text_mode not in {"content", "contextual"}:
        raise ValueError(
            "text_mode chỉ nhận 'content' hoặc 'contextual'."
        )

    return available_columns, row_count


def export_lookup_table(
    input_path: Path,
    output_path: Path,
    available_columns: list[str],
    read_batch_size: int,
) -> int:
    """
    Lưu bảng ánh xạ bm25_row sang chunk gốc theo kiểu streaming.
    Không nạp toàn bộ parquet vào RAM.
    """
    selected_columns = [
        column
        for column in LOOKUP_COLUMNS
        if column in available_columns
    ]

    parquet_file = pq.ParquetFile(input_path)
    writer: pq.ParquetWriter | None = None
    current_row = 0

    try:
        for batch in parquet_file.iter_batches(
            batch_size=read_batch_size,
            columns=selected_columns,
            use_threads=True,
        ):
            table = pa.Table.from_batches([batch])
            batch_rows = table.num_rows

            bm25_rows = pa.array(
                np.arange(
                    current_row,
                    current_row + batch_rows,
                    dtype=np.int64,
                )
            )

            table = table.add_column(
                0,
                "bm25_row",
                bm25_rows,
            )

            if writer is None:
                writer = pq.ParquetWriter(
                    output_path,
                    table.schema,
                    compression="zstd",
                )

            writer.write_table(table)
            current_row += batch_rows

    finally:
        if writer is not None:
            writer.close()

    LOGGER.info(
        "Đã lưu bảng tra cứu: %s (%s dòng)",
        output_path,
        f"{current_row:,}",
    )

    return current_row


def tokenize_parquet(
    input_path: Path,
    tokenizer: Any,
    text_columns: list[str],
    text_mode: str,
    read_batch_size: int,
    expected_rows: int,
) -> tuple[Any, dict[str, Any]]:
    """
    Đọc parquet theo batch và token hóa từng batch.

    Chỉ giữ token ID trong RAM, không giữ toàn bộ corpus text.
    Vocabulary được mở rộng liên tục giữa các batch.
    """
    parquet_file = pq.ParquetFile(input_path)
    corpus_token_ids: list[list[int]] = []

    processed_rows = 0
    total_tokens = 0
    empty_documents = 0
    started_at = time.time()

    for record_batch in parquet_file.iter_batches(
        batch_size=read_batch_size,
        columns=text_columns,
        use_threads=True,
    ):
        texts = batch_to_texts(
            batch=record_batch,
            text_mode=text_mode,
        )

        tokenized_batch = tokenizer.tokenize(
            texts,
            update_vocab=True,
            return_as="ids",
        )

        if hasattr(tokenized_batch, "ids"):
            batch_ids = tokenized_batch.ids
        else:
            batch_ids = tokenized_batch

        for token_ids in batch_ids:
            ids = [int(token_id) for token_id in token_ids]

            if not ids:
                empty_documents += 1

            total_tokens += len(ids)
            corpus_token_ids.append(ids)

        processed_rows += len(batch_ids)

        elapsed = max(time.time() - started_at, 1e-9)
        speed = processed_rows / elapsed

        LOGGER.info(
            "Tokenized %s/%s chunk (%.2f%%) | %.2f chunk/giây",
            f"{processed_rows:,}",
            f"{expected_rows:,}",
            100.0 * processed_rows / expected_rows,
            speed,
        )

    if processed_rows != expected_rows:
        raise RuntimeError(
            "Số tài liệu token hóa không khớp:\n"
            f"tokenized = {processed_rows:,}\n"
            f"expected  = {expected_rows:,}"
        )

    tokenized_corpus = tokenizer.to_tokenized_tuple(
        corpus_token_ids
    )

    statistics = {
        "row_count": int(processed_rows),
        "total_tokens": int(total_tokens),
        "mean_tokens_per_chunk": (
            float(total_tokens / processed_rows)
            if processed_rows
            else 0.0
        ),
        "empty_documents": int(empty_documents),
        "vocabulary_size": int(
            len(tokenizer.get_vocab_dict())
        ),
        "tokenization_seconds": float(
            time.time() - started_at
        ),
    }

    return tokenized_corpus, statistics


def run_test_query(
    retriever: Any,
    tokenizer: Any,
    query: str | None,
    top_k: int,
    row_count: int,
) -> dict[str, Any] | None:
    if not query:
        return None

    query_tokens = tokenizer.tokenize(
        [query],
        update_vocab=False,
        return_as="ids",
    )

    if hasattr(query_tokens, "ids"):
        query_tokens = query_tokens.ids

    effective_k = min(
        max(1, top_k),
        row_count,
    )

    result_ids, scores = retriever.retrieve(
        query_tokens,
        k=effective_k,
    )

    ids_list = (
        np.asarray(result_ids)[0]
        .astype(np.int64)
        .tolist()
    )

    scores_list = (
        np.asarray(scores)[0]
        .astype(float)
        .tolist()
    )

    LOGGER.info("Test query: %s", query)
    LOGGER.info("Top BM25 rows: %s", ids_list)
    LOGGER.info("Top BM25 scores: %s", scores_list)

    return {
        "query": query,
        "top_k": int(effective_k),
        "bm25_rows": ids_list,
        "scores": scores_list,
    }


def build_bm25_index(
    input_path: str,
    output_dir: str,
    text_mode: str = "contextual",
    read_batch_size: int = 512,
    method: str = "lucene",
    k1: float = 1.5,
    b: float = 0.75,
    delta: float = 0.5,
    backend: str = "numpy",
    csc_backend: str = "numpy",
    splitter: str = r"(?u)\b\w+\b",
    stopwords: list[str] | None = None,
    save_lookup_table: bool = True,
    test_query: str | None = None,
    test_top_k: int = 5,
    overwrite: bool = True,
    log_level: str = "INFO",
) -> None:
    """
    Xây BM25 index bằng bm25s.

    text_mode:
        "content"    : chỉ index cột content.
        "contextual" : index số hiệu, tiêu đề, cấu trúc và content.

    Lưu ý:
    - Không loại stopword mặc định vì các từ như "không", "phải",
      "được" có ý nghĩa quan trọng trong văn bản pháp luật.
    - bm25_row giữ cùng thứ tự với dòng parquet và FAISS ID.
    """
    configure_logging(log_level)

    if read_batch_size <= 0:
        raise ValueError("read_batch_size phải lớn hơn 0.")

    if k1 <= 0:
        raise ValueError("k1 phải lớn hơn 0.")

    if not 0.0 <= b <= 1.0:
        raise ValueError("b phải nằm trong [0, 1].")

    if method not in {
        "lucene",
        "robertson",
        "atire",
        "bm25l",
        "bm25+",
    }:
        raise ValueError(
            "method không hợp lệ. Dùng lucene, robertson, "
            "atire, bm25l hoặc bm25+."
        )

    if backend not in {"numpy", "numba", "auto"}:
        raise ValueError(
            "backend chỉ nhận numpy, numba hoặc auto."
        )

    if csc_backend not in {"numpy", "scipy", "auto"}:
        raise ValueError(
            "csc_backend chỉ nhận numpy, scipy hoặc auto."
        )

    input_path_obj = Path(input_path)
    output_dir_obj = Path(output_dir)
    lookup_path = output_dir_obj / "bm25_lookup.parquet"
    manifest_path = output_dir_obj / "bm25_manifest.json"

    available_columns, row_count = validate_input(
        input_path=input_path_obj,
        text_mode=text_mode,
    )

    if output_dir_obj.exists():
        if overwrite:
            LOGGER.warning(
                "Xóa BM25 output cũ: %s",
                output_dir_obj.resolve(),
            )
            shutil.rmtree(output_dir_obj)
        elif any(output_dir_obj.iterdir()):
            raise FileExistsError(
                f"Output đã tồn tại: {output_dir_obj.resolve()}"
            )

    output_dir_obj.mkdir(
        parents=True,
        exist_ok=True,
    )

    text_columns = (
        ["content"]
        if text_mode == "content"
        else [
            column
            for column in TEXT_COLUMNS
            if column in available_columns
        ]
    )

    bm25s, Tokenizer = require_bm25s()

    actual_stopwords = (
        []
        if stopwords is None
        else stopwords
    )

    tokenizer = Tokenizer(
        lower=True,
        stopwords=actual_stopwords,
        splitter=splitter,
    )

    LOGGER.info("Input: %s", input_path_obj.resolve())
    LOGGER.info("Số chunk: %s", f"{row_count:,}")
    LOGGER.info("Text mode: %s", text_mode)
    LOGGER.info("Các cột index: %s", text_columns)
    LOGGER.info("BM25 method: %s", method)
    LOGGER.info("k1=%s | b=%s | delta=%s", k1, b, delta)
    LOGGER.info(
        "backend=%s | csc_backend=%s",
        backend,
        csc_backend,
    )

    tokenized_corpus, token_statistics = tokenize_parquet(
        input_path=input_path_obj,
        tokenizer=tokenizer,
        text_columns=text_columns,
        text_mode=text_mode,
        read_batch_size=read_batch_size,
        expected_rows=row_count,
    )

    LOGGER.info(
        "Vocabulary size: %s",
        f"{token_statistics['vocabulary_size']:,}",
    )
    LOGGER.info(
        "Tổng token: %s",
        f"{token_statistics['total_tokens']:,}",
    )
    LOGGER.info("Bắt đầu xây BM25 sparse index...")

    index_started_at = time.time()

    retriever = bm25s.BM25(
        k1=k1,
        b=b,
        delta=delta,
        method=method,
        backend=backend,
        csc_backend=csc_backend,
    )

    retriever.index(
        tokenized_corpus,
        show_progress=True,
        leave_progress=True,
    )

    index_seconds = time.time() - index_started_at

    LOGGER.info(
        "Xây BM25 index xong trong %.2f giây.",
        index_seconds,
    )

    test_result = run_test_query(
        retriever=retriever,
        tokenizer=tokenizer,
        query=test_query,
        top_k=test_top_k,
        row_count=row_count,
    )

    LOGGER.info(
        "Đang lưu BM25 index vào: %s",
        output_dir_obj.resolve(),
    )

    retriever.save(
        str(output_dir_obj),
    )

    tokenizer.save_vocab(
        save_dir=str(output_dir_obj),
    )

    tokenizer.save_stopwords(
        save_dir=str(output_dir_obj),
    )

    lookup_rows: int | None = None

    if save_lookup_table:
        lookup_rows = export_lookup_table(
            input_path=input_path_obj,
            output_path=lookup_path,
            available_columns=available_columns,
            read_batch_size=read_batch_size,
        )

        if lookup_rows != row_count:
            raise RuntimeError(
                "Số dòng lookup không khớp với BM25 index."
            )

    manifest = {
        "input_path": str(input_path_obj.resolve()),
        "output_dir": str(output_dir_obj.resolve()),
        "lookup_path": (
            str(lookup_path.resolve())
            if save_lookup_table
            else None
        ),
        "bm25s_version": getattr(
            bm25s,
            "__version__",
            "unknown",
        ),
        "row_count": int(row_count),
        "lookup_rows": lookup_rows,
        "row_alignment": (
            "bm25_row == FAISS id == source parquet row"
        ),
        "text_mode": text_mode,
        "text_columns": text_columns,
        "tokenizer": {
            "lower": True,
            "splitter": splitter,
            "stopwords": actual_stopwords,
            "stemming": None,
        },
        "bm25": {
            "method": method,
            "k1": float(k1),
            "b": float(b),
            "delta": float(delta),
            "backend": backend,
            "csc_backend": csc_backend,
        },
        "token_statistics": token_statistics,
        "index_seconds": float(index_seconds),
        "test_result": test_result,
        "created_at": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    }

    save_json(
        manifest_path,
        manifest,
    )

    print("\n" + "=" * 72)
    print("BM25 INDEX COMPLETED")
    print("=" * 72)
    print(f"Input       : {input_path_obj.resolve()}")
    print(f"Rows        : {row_count:,}")
    print(f"Vocabulary  : {token_statistics['vocabulary_size']:,}")
    print(f"Total tokens: {token_statistics['total_tokens']:,}")
    print(f"Text mode   : {text_mode}")
    print(f"Method      : {method}")
    print(f"Index dir   : {output_dir_obj.resolve()}")
    print(f"Lookup      : {lookup_path.resolve() if save_lookup_table else 'Không lưu'}")
    print(f"Manifest    : {manifest_path.resolve()}")
    print(
        "Row mapping : "
        "bm25_row == FAISS id == source parquet row"
    )


def search_saved_bm25(
    index_dir: str,
    query: str,
    top_k: int = 10,
    mmap: bool = True,
    splitter: str = r"(?u)\b\w+\b",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Hàm tìm kiếm thủ công để kiểm tra index đã lưu.

    Trả về:
        bm25_rows: ID dòng, dùng trực tiếp để fusion với FAISS.
        scores: điểm BM25.
    """
    bm25s, Tokenizer = require_bm25s()
    index_dir_obj = Path(index_dir)

    if not index_dir_obj.exists():
        raise FileNotFoundError(
            f"Không tìm thấy BM25 index: {index_dir_obj.resolve()}"
        )

    retriever = bm25s.BM25.load(
        str(index_dir_obj),
        mmap=mmap,
        load_corpus=False,
    )

    tokenizer = Tokenizer(
        lower=True,
        stopwords=[],
        splitter=splitter,
    )

    tokenizer.load_vocab(
        str(index_dir_obj)
    )

    tokenizer.load_stopwords(
        str(index_dir_obj)
    )

    query_tokens = tokenizer.tokenize(
        [query],
        update_vocab=False,
        return_as="ids",
    )

    if hasattr(query_tokens, "ids"):
        query_tokens = query_tokens.ids

    result_ids, scores = retriever.retrieve(
        query_tokens,
        k=top_k,
    )

    bm25_rows = (
        np.asarray(result_ids)[0]
        .astype(np.int64)
    )

    bm25_scores = (
        np.asarray(scores)[0]
        .astype(np.float32)
    )

    print("\nBM25 QUERY:", query)

    for rank, (row_id, score) in enumerate(
        zip(bm25_rows, bm25_scores),
        start=1,
    ):
        print(
            f"{rank:02d}. bm25_row={int(row_id):,} "
            f"| score={float(score):.6f}"
        )

    return bm25_rows, bm25_scores


# ============================================================
# GỌI HÀM THỦ CÔNG TẠI ĐÂY
# KHÔNG DÙNG ARGPARSE VÀ KHÔNG DÙNG MAIN
# ============================================================

build_bm25_index(
    input_path="/workspace/legal_rag/legal_chunks.parquet",
    output_dir="/workspace/legal_rag/bm25_output",

    # "content": chỉ index nội dung.
    # "contextual": index số hiệu + tiêu đề + phần/chương/mục/điều + nội dung.
    text_mode="contextual",

    # Số chunk đọc và token hóa mỗi lượt.
    read_batch_size=512,

    # Cấu hình BM25.
    method="lucene",
    k1=1.5,
    b=0.75,
    delta=0.5,

    # CPU backend, không cần GPU.
    backend="numpy",
    csc_backend="numpy",

    # Giữ từ tiếng Việt có dấu, số điều, số văn bản.
    splitter=r"(?u)\b\w+\b",

    # Không bỏ stopword pháp lý như: không, phải, được.
    stopwords=[],

    # Lưu bm25_lookup.parquet để lấy nội dung theo bm25_row.
    save_lookup_table=True,

    # Query kiểm tra sau khi build; đặt None nếu không muốn test.
    test_query="quy định về thủ tục hành chính",
    test_top_k=5,

    overwrite=True,
    log_level="INFO",
)


# Sau khi index đã xây xong, có thể comment build_bm25_index(...) ở trên
# và bỏ comment đoạn dưới để tìm kiếm thủ công:
#
# search_saved_bm25(
#     index_dir="/workspace/legal_rag/bm25_output",
#     query="điều kiện cấp giấy chứng nhận quyền sử dụng đất",
#     top_k=10,
#     mmap=True,
# )
