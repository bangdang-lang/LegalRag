"""
build_faiss_index.py

Xây FAISS index từ embeddings.npy.

File này KHÔNG chạy embedding model.
Nó chỉ:
1. Đọc embeddings.npy.
2. Chuẩn hóa vector nếu cần.
3. Xây FAISS index.
4. Lưu index và manifest.

Đầu vào mặc định:
    embedding_output/embeddings.npy
    embedding_output/embedding_metadata.parquet

Đầu ra mặc định:
    faiss_output/legal_chunks.faiss
    faiss_output/faiss_manifest.json

Cài đặt:
    pip install numpy pyarrow faiss-cpu

Xây exact cosine index:
    python build_faiss_index.py

Xây HNSW index:
    python build_faiss_index.py \
        --index-type hnsw \
        --hnsw-m 32 \
        --ef-construction 200
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


LOGGER = logging.getLogger("faiss_builder")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Xây FAISS index từ embeddings.npy."
    )

    parser.add_argument(
        "--embeddings",
        default="embedding_output/embeddings.npy",
        help="Đường dẫn tới embeddings.npy.",
    )
    parser.add_argument(
        "--metadata",
        default="embedding_output/embedding_metadata.parquet",
        help=(
            "Metadata dùng để kiểm tra số dòng và ánh xạ kết quả. "
            "Có thể bỏ qua bằng --skip-metadata-check."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="faiss_output",
        help="Thư mục lưu FAISS index.",
    )
    parser.add_argument(
        "--index-name",
        default="legal_chunks.faiss",
        help="Tên file FAISS index.",
    )
    parser.add_argument(
        "--index-type",
        choices=["flat", "hnsw"],
        default="flat",
        help=(
            "flat: tìm kiếm chính xác; "
            "hnsw: gần đúng, nhanh hơn với dữ liệu lớn."
        ),
    )
    parser.add_argument(
        "--metric",
        choices=["cosine", "l2"],
        default="cosine",
        help="Metric tìm kiếm. Mặc định: cosine.",
    )
    parser.add_argument(
        "--add-batch-size",
        type=int,
        default=10_000,
        help="Số vector thêm vào index mỗi lượt.",
    )
    parser.add_argument(
        "--hnsw-m",
        type=int,
        default=32,
        help="Số kết nối HNSW trên mỗi node.",
    )
    parser.add_argument(
        "--ef-construction",
        type=int,
        default=200,
        help="Độ chính xác khi xây HNSW.",
    )
    parser.add_argument(
        "--ef-search",
        type=int,
        default=128,
        help="Độ chính xác mặc định khi tìm kiếm HNSW.",
    )
    parser.add_argument(
        "--skip-normalize",
        action="store_true",
        help=(
            "Không chuẩn hóa L2 trước khi thêm index. "
            "Chỉ nên dùng khi vector đã chắc chắn chuẩn hóa."
        ),
    )
    parser.add_argument(
        "--skip-metadata-check",
        action="store_true",
        help="Không kiểm tra số dòng của metadata.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ghi đè index đã tồn tại.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )

    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def require_faiss() -> Any:
    try:
        import faiss
    except ImportError as error:
        raise RuntimeError(
            "Thiếu faiss-cpu. Cài bằng: pip install faiss-cpu"
        ) from error

    return faiss


def save_json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    os.replace(temporary, path)


def load_embeddings(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy embedding: {path.resolve()}"
        )

    embeddings = np.load(path, mmap_mode="r")

    if embeddings.ndim != 2:
        raise ValueError(
            f"embeddings.npy phải có 2 chiều, nhận được {embeddings.shape}."
        )

    if embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
        raise ValueError("embeddings.npy rỗng.")

    if not np.issubdtype(embeddings.dtype, np.floating):
        raise TypeError(
            f"Embedding phải là số thực, nhận được dtype={embeddings.dtype}."
        )

    return embeddings


def check_metadata(
    metadata_path: Path,
    embedding_rows: int,
) -> int:
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy metadata: {metadata_path.resolve()}"
        )

    parquet_file = pq.ParquetFile(metadata_path)
    metadata_rows = parquet_file.metadata.num_rows

    if metadata_rows != embedding_rows:
        raise ValueError(
            "Số dòng metadata không bằng số embedding: "
            f"metadata={metadata_rows:,}, embeddings={embedding_rows:,}."
        )

    return metadata_rows


def create_base_index(
    faiss: Any,
    dimension: int,
    index_type: str,
    metric: str,
    hnsw_m: int,
    ef_construction: int,
    ef_search: int,
) -> Any:
    metric_type = (
        faiss.METRIC_INNER_PRODUCT
        if metric == "cosine"
        else faiss.METRIC_L2
    )

    if index_type == "flat":
        if metric == "cosine":
            base_index = faiss.IndexFlatIP(dimension)
        else:
            base_index = faiss.IndexFlatL2(dimension)

    else:
        if hnsw_m <= 0:
            raise ValueError("--hnsw-m phải lớn hơn 0.")

        base_index = faiss.IndexHNSWFlat(
            dimension,
            hnsw_m,
            metric_type,
        )
        base_index.hnsw.efConstruction = ef_construction
        base_index.hnsw.efSearch = ef_search

    # ID trong FAISS chính là embedding_row.
    return faiss.IndexIDMap2(base_index)


def validate_sample(embeddings: np.ndarray) -> dict[str, float]:
    row_count = embeddings.shape[0]
    sample_count = min(row_count, 1000)
    indices = np.linspace(
        0,
        row_count - 1,
        num=sample_count,
        dtype=np.int64,
    )

    sample = np.asarray(embeddings[indices], dtype=np.float32)

    if not np.isfinite(sample).all():
        raise ValueError("Embedding chứa NaN hoặc Inf.")

    norms = np.linalg.norm(sample, axis=1)

    return {
        "sample_count": int(sample_count),
        "minimum_norm_before_index": float(norms.min()),
        "mean_norm_before_index": float(norms.mean()),
        "maximum_norm_before_index": float(norms.max()),
    }


def add_embeddings(
    faiss: Any,
    index: Any,
    embeddings: np.ndarray,
    batch_size: int,
    normalize: bool,
) -> None:
    row_count = embeddings.shape[0]

    if batch_size <= 0:
        raise ValueError("--add-batch-size phải lớn hơn 0.")

    started = time.time()

    for start in range(0, row_count, batch_size):
        end = min(start + batch_size, row_count)

        vectors = np.ascontiguousarray(
            embeddings[start:end],
            dtype=np.float32,
        )

        if not np.isfinite(vectors).all():
            raise ValueError(
                f"Phát hiện NaN hoặc Inf trong khoảng dòng [{start}, {end})."
            )

        if normalize:
            faiss.normalize_L2(vectors)

        ids = np.arange(start, end, dtype=np.int64)
        index.add_with_ids(vectors, ids)

        elapsed = max(time.time() - started, 1e-9)
        speed = end / elapsed

        LOGGER.info(
            "Đã thêm %s/%s vector (%.2f%%) | %.2f vector/giây",
            f"{end:,}",
            f"{row_count:,}",
            100.0 * end / row_count,
            speed,
        )


def verify_index(
    index: Any,
    embeddings: np.ndarray,
    normalize: bool,
    faiss: Any,
) -> dict[str, Any]:
    if index.ntotal != embeddings.shape[0]:
        raise RuntimeError(
            f"Index chứa {index.ntotal:,} vector, "
            f"nhưng embedding có {embeddings.shape[0]:,} dòng."
        )

    probe_count = min(5, embeddings.shape[0])
    queries = np.ascontiguousarray(
        embeddings[:probe_count],
        dtype=np.float32,
    )

    if normalize:
        faiss.normalize_L2(queries)

    scores, ids = index.search(queries, 1)
    expected_ids = np.arange(probe_count, dtype=np.int64)
    returned_ids = ids[:, 0]

    return {
        "ntotal": int(index.ntotal),
        "probe_count": int(probe_count),
        "self_retrieval_ids": returned_ids.tolist(),
        "expected_ids": expected_ids.tolist(),
        "self_retrieval_passed": bool(
            np.array_equal(returned_ids, expected_ids)
        ),
        "self_retrieval_scores": scores[:, 0].astype(float).tolist(),
    }


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    embeddings_path = Path(args.embeddings)
    metadata_path = Path(args.metadata)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = output_dir / args.index_name
    manifest_path = output_dir / "faiss_manifest.json"

    if index_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Index đã tồn tại: {index_path}. "
            "Dùng --overwrite để ghi đè."
        )

    faiss = require_faiss()
    embeddings = load_embeddings(embeddings_path)
    row_count, dimension = embeddings.shape

    LOGGER.info("Embeddings: %s", embeddings_path.resolve())
    LOGGER.info("Shape: (%s, %s)", f"{row_count:,}", dimension)
    LOGGER.info("Index type: %s", args.index_type)
    LOGGER.info("Metric: %s", args.metric)

    metadata_rows: int | None = None

    if not args.skip_metadata_check:
        metadata_rows = check_metadata(
            metadata_path=metadata_path,
            embedding_rows=row_count,
        )
        LOGGER.info(
            "Metadata hợp lệ: %s dòng",
            f"{metadata_rows:,}",
        )

    sample_validation = validate_sample(embeddings)

    index = create_base_index(
        faiss=faiss,
        dimension=dimension,
        index_type=args.index_type,
        metric=args.metric,
        hnsw_m=args.hnsw_m,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
    )

    # Cosine similarity = inner product sau khi chuẩn hóa L2.
    normalize = args.metric == "cosine" and not args.skip_normalize

    add_embeddings(
        faiss=faiss,
        index=index,
        embeddings=embeddings,
        batch_size=args.add_batch_size,
        normalize=normalize,
    )

    verification = verify_index(
        index=index,
        embeddings=embeddings,
        normalize=normalize,
        faiss=faiss,
    )

    temporary_index_path = index_path.with_suffix(
        index_path.suffix + ".tmp"
    )
    faiss.write_index(index, str(temporary_index_path))
    os.replace(temporary_index_path, index_path)

    manifest = {
        "embeddings_path": str(embeddings_path.resolve()),
        "metadata_path": (
            str(metadata_path.resolve())
            if not args.skip_metadata_check
            else None
        ),
        "index_path": str(index_path.resolve()),
        "index_type": args.index_type,
        "metric": args.metric,
        "normalized_before_add": normalize,
        "row_count": int(row_count),
        "dimension": int(dimension),
        "dtype_source": str(embeddings.dtype),
        "metadata_rows": metadata_rows,
        "hnsw_m": args.hnsw_m if args.index_type == "hnsw" else None,
        "ef_construction": (
            args.ef_construction
            if args.index_type == "hnsw"
            else None
        ),
        "ef_search": (
            args.ef_search
            if args.index_type == "hnsw"
            else None
        ),
        "sample_validation": sample_validation,
        "verification": verification,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json(manifest_path, manifest)

    print("\n" + "=" * 72)
    print("FAISS INDEX COMPLETED")
    print("=" * 72)
    print(f"Index type  : {args.index_type}")
    print(f"Metric      : {args.metric}")
    print(f"Vectors     : {row_count:,}")
    print(f"Dimension   : {dimension}")
    print(f"Index       : {index_path}")
    print(f"Manifest    : {manifest_path}")
    print(
        "Self-test   : "
        f"{'PASSED' if verification['self_retrieval_passed'] else 'WARNING'}"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOGGER.warning("Đã dừng xây FAISS index.")
        sys.exit(130)
    except Exception as error:
        LOGGER.exception("Xây FAISS index thất bại: %s", error)
        sys.exit(1)
