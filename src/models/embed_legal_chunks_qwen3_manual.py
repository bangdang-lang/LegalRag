"""
embed_legal_chunks_qwen3.py

Sinh embedding cho legal_chunks.parquet bằng:
    Qwen/Qwen3-Embedding-0.6B

File này CHỈ làm nhiệm vụ embedding.
FAISS được xây riêng bằng build_faiss_index.py.

Đầu ra:
    embedding_output/
    ├── embeddings.npy
    ├── embedding_metadata.parquet
    ├── embedding_config.json
    └── embedding_state.json

Cài đặt:
    pip install "transformers>=4.51.0" "sentence-transformers>=2.7.0" \
        torch numpy pyarrow

Cách chạy:
1. Sửa các tham số trong khối run_embedding(...) ở cuối file.
2. Chạy:
       python embed_legal_chunks_qwen3_manual.py

Không dùng argparse hoặc tham số command line.

Lưu ý:
- Document embedding KHÔNG dùng query instruction.
- Embedding được chuẩn hóa L2 để dùng cosine similarity qua FAISS Inner Product.
- File embeddings.npy được ghi bằng memmap, không giữ toàn bộ vector trong RAM.
- Có checkpoint. Nếu bị dừng, chạy lại cùng lệnh để tiếp tục.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


LOGGER = logging.getLogger("qwen3_legal_embedding")

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_INPUT = "./data/legal_chunks.parquet"
DEFAULT_OUTPUT_DIR = "embedding_output"

# Các cột được dùng để tạo chuỗi embedding.
TEXT_COLUMNS = [
    "document_number",
    "title",
    "part",
    "chapter",
    "section",
    "articles",
    "content",
]

# Metadata giữ lại để ánh xạ embedding_row -> chunk.
METADATA_COLUMNS = [
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
]


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def require_dependencies() -> tuple[Any, Any]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "Thiếu torch. Cài bằng: pip install torch"
        ) from error

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "Thiếu sentence-transformers. Cài bằng: "
            'pip install "sentence-transformers>=2.7.0" '
            '"transformers>=4.51.0"'
        ) from error

    return torch, SentenceTransformer


def choose_device(requested: str, torch_module: Any) -> str:
    if requested == "cuda":
        if not torch_module.cuda.is_available():
            raise RuntimeError("PyTorch không phát hiện GPU CUDA.")
        return "cuda"

    if requested == "mps":
        available = (
            hasattr(torch_module.backends, "mps")
            and torch_module.backends.mps.is_available()
        )
        if not available:
            raise RuntimeError("Thiết bị hiện tại không hỗ trợ MPS.")
        return "mps"

    if requested == "cpu":
        return "cpu"

    if torch_module.cuda.is_available():
        return "cuda"

    mps_available = (
        hasattr(torch_module.backends, "mps")
        and torch_module.backends.mps.is_available()
    )
    if mps_available:
        return "mps"

    return "cpu"


def choose_batch_size(device: str, requested: int | None) -> int:
    if requested is not None:
        if requested <= 0:
            raise ValueError("--batch-size phải lớn hơn 0.")
        return requested

    return 2 if device == "cpu" else 16


def safe_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return ""

    text = str(value).strip()

    if text.lower() in {"", "none", "null", "nan"}:
        return ""

    return text


def build_contextual_text(record: dict[str, Any]) -> str:
    """
    Ghép cấu trúc văn bản vào nội dung để vector giữ được ngữ cảnh pháp luật.
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
        parts.append(f"Nội dung:\n{content}")

    text = "\n".join(parts).strip()
    return text or "[EMPTY LEGAL CHUNK]"


def batch_to_texts(
    batch: pa.RecordBatch,
    text_mode: str,
) -> list[str]:
    columns = batch.to_pydict()
    texts: list[str] = []

    for row_index in range(batch.num_rows):
        if text_mode == "content":
            text = safe_text(columns["content"][row_index])
            texts.append(text or "[EMPTY LEGAL CHUNK]")
            continue

        record = {
            column: values[row_index]
            for column, values in columns.items()
        }
        texts.append(build_contextual_text(record))

    return texts


def validate_input(path: Path) -> tuple[list[str], int]:
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file: {path.resolve()}\n"
            f"Thư mục chạy hiện tại: {Path.cwd()}"
        )

    parquet_file = pq.ParquetFile(path)
    columns = parquet_file.schema_arrow.names

    missing = [
        required
        for required in ["id", "content"]
        if required not in columns
    ]

    if missing:
        raise ValueError(
            f"File thiếu cột bắt buộc: {missing}. "
            f"Các cột hiện có: {columns}"
        )

    return columns, parquet_file.metadata.num_rows


def save_json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def export_metadata(
    input_path: Path,
    output_path: Path,
    available_columns: list[str],
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        LOGGER.info("Giữ metadata đã có: %s", output_path)
        return

    selected = [
        column
        for column in METADATA_COLUMNS
        if column in available_columns
    ]

    table = pq.read_table(input_path, columns=selected)
    embedding_rows = pa.array(
        np.arange(table.num_rows, dtype=np.int64)
    )

    table = table.add_column(
        0,
        "embedding_row",
        embedding_rows,
    )

    pq.write_table(
        table,
        output_path,
        compression="zstd",
    )

    LOGGER.info(
        "Đã lưu metadata: %s (%s dòng)",
        output_path,
        f"{table.num_rows:,}",
    )


def load_model(
    model_name: str,
    device: str,
    cache_dir: str | None,
    local_files_only: bool,
    max_seq_length: int,
    SentenceTransformer: Any,
) -> Any:
    kwargs: dict[str, Any] = {
        "device": device,
        "local_files_only": local_files_only,
    }

    if cache_dir:
        kwargs["cache_folder"] = cache_dir

    LOGGER.info("Đang load model: %s", model_name)
    model = SentenceTransformer(model_name, **kwargs)
    model.max_seq_length = max_seq_length

    return model


def detect_full_dimension(model: Any, device: str) -> int:
    dimension = model.get_sentence_embedding_dimension()

    if dimension is not None:
        return int(dimension)

    probe = model.encode(
        ["Kiểm tra kích thước embedding"],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        device=device,
    )

    return int(probe.shape[1])


def truncate_and_normalize(
    vectors: np.ndarray,
    target_dimension: int,
) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)

    if vectors.ndim != 2:
        raise RuntimeError(
            f"Embedding phải có 2 chiều, nhận được {vectors.shape}."
        )

    if target_dimension > vectors.shape[1]:
        raise ValueError(
            f"embedding_dim={target_dimension} lớn hơn dimension "
            f"của model là {vectors.shape[1]}."
        )

    if target_dimension < vectors.shape[1]:
        vectors = vectors[:, :target_dimension]

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)

    return np.asarray(vectors / norms, dtype=np.float32)


def prepare_storage(
    embeddings_path: Path,
    state_path: Path,
    row_count: int,
    embedding_dim: int,
    model_name: str,
    input_path: Path,
    overwrite: bool,
) -> tuple[np.memmap, dict[str, Any]]:
    if overwrite:
        embeddings_path.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)

    previous_state = load_json(state_path)

    if embeddings_path.exists() and previous_state is not None:
        expected = {
            "row_count": row_count,
            "embedding_dim": embedding_dim,
            "model": model_name,
        }

        for key, expected_value in expected.items():
            actual_value = previous_state.get(key)

            if actual_value != expected_value:
                raise RuntimeError(
                    "Checkpoint không khớp cấu hình hiện tại. "
                    "Dùng --overwrite để chạy lại.\n"
                    f"{key}: cũ={actual_value!r}, mới={expected_value!r}"
                )

        embeddings = np.lib.format.open_memmap(
            embeddings_path,
            mode="r+",
            dtype=np.float32,
            shape=(row_count, embedding_dim),
        )

        return embeddings, previous_state

    embeddings = np.lib.format.open_memmap(
        embeddings_path,
        mode="w+",
        dtype=np.float32,
        shape=(row_count, embedding_dim),
    )

    state = {
        "input_path": str(input_path.resolve()),
        "model": model_name,
        "row_count": row_count,
        "embedding_dim": embedding_dim,
        "next_row": 0,
        "completed": False,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    save_json(state_path, state)
    return embeddings, state


def embed_parquet(
    input_path: Path,
    model: Any,
    embeddings: np.memmap,
    state: dict[str, Any],
    state_path: Path,
    text_columns: list[str],
    text_mode: str,
    batch_size: int,
    read_batch_size: int,
    target_dimension: int,
    device: str,
) -> None:
    parquet_file = pq.ParquetFile(input_path)
    total_rows = parquet_file.metadata.num_rows
    next_row = int(state.get("next_row", 0))
    resume_from = next_row

    if next_row >= total_rows:
        LOGGER.info("Embedding đã hoàn tất từ lần chạy trước.")
        return

    LOGGER.info(
        "Bắt đầu từ dòng %s/%s",
        f"{next_row:,}",
        f"{total_rows:,}",
    )

    global_start = 0
    started = time.time()

    for record_batch in parquet_file.iter_batches(
        batch_size=read_batch_size,
        columns=text_columns,
        use_threads=True,
    ):
        original_batch_size = record_batch.num_rows
        global_end = global_start + original_batch_size

        if global_end <= next_row:
            global_start = global_end
            continue

        local_start = max(0, next_row - global_start)

        if local_start:
            record_batch = record_batch.slice(
                local_start,
                original_batch_size - local_start,
            )

        texts = batch_to_texts(record_batch, text_mode)

        output_start = global_start + local_start
        output_end = output_start + len(texts)

        # Đây là document embedding, không truyền prompt_name="query".
        full_vectors = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            device=device,
        )

        vectors = truncate_and_normalize(
            full_vectors,
            target_dimension=target_dimension,
        )

        embeddings[output_start:output_end] = vectors
        embeddings.flush()

        next_row = output_end
        state["next_row"] = next_row
        state["completed"] = next_row >= total_rows
        state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save_json(state_path, state)

        elapsed = max(time.time() - started, 1e-9)
        processed = next_row - resume_from
        speed = processed / elapsed

        LOGGER.info(
            "Đã xử lý %s/%s (%.2f%%) | %.2f chunk/giây",
            f"{next_row:,}",
            f"{total_rows:,}",
            100.0 * next_row / total_rows,
            speed,
        )

        global_start = global_end

    state["completed"] = True
    state["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_json(state_path, state)


def validate_result(embeddings: np.memmap) -> dict[str, Any]:
    row_count = embeddings.shape[0]
    sample_count = min(1000, row_count)

    indices = np.linspace(
        0,
        row_count - 1,
        num=sample_count,
        dtype=np.int64,
    )

    sample = np.asarray(embeddings[indices], dtype=np.float32)

    if not np.isfinite(sample).all():
        raise RuntimeError("Embedding chứa NaN hoặc Inf.")

    norms = np.linalg.norm(sample, axis=1)

    return {
        "sample_count": int(sample_count),
        "minimum_norm": float(norms.min()),
        "mean_norm": float(norms.mean()),
        "maximum_norm": float(norms.max()),
    }



def run_embedding(
    input_path: str = DEFAULT_INPUT,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    model_name: str = DEFAULT_MODEL,
    device_request: str = "cpu",
    batch_size_request: int | None = 2,
    read_batch_size: int = 128,
    max_seq_length: int = 512,
    embedding_dim: int = 1024,
    text_mode: str = "content",
    cache_dir: str | None = None,
    local_files_only: bool = False,
    overwrite: bool = False,
    log_level: str = "INFO",
) -> None:
    """
    Chạy embedding bằng cách truyền tham số trực tiếp.

    Ví dụ:
        run_embedding(
            input_path="./data/legal_chunks.parquet",
            output_dir="./embedding_output",
            device_request="cpu",
            batch_size_request=4,
            max_seq_length=512,
            text_mode="content",
        )
    """
    configure_logging(log_level)

    if read_batch_size <= 0:
        raise ValueError("read_batch_size phải lớn hơn 0.")

    if max_seq_length <= 0:
        raise ValueError("max_seq_length phải lớn hơn 0.")

    if not 32 <= embedding_dim <= 1024:
        raise ValueError("embedding_dim phải nằm trong [32, 1024].")

    if text_mode not in {"content", "contextual"}:
        raise ValueError(
            "text_mode chỉ nhận 'content' hoặc 'contextual'."
        )

    if device_request not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(
            "device_request chỉ nhận auto, cpu, cuda hoặc mps."
        )

    input_path_obj = Path(input_path)
    output_dir_obj = Path(output_dir)
    output_dir_obj.mkdir(parents=True, exist_ok=True)

    embeddings_path = output_dir_obj / "embeddings.npy"
    metadata_path = output_dir_obj / "embedding_metadata.parquet"
    config_path = output_dir_obj / "embedding_config.json"
    state_path = output_dir_obj / "embedding_state.json"

    available_columns, row_count = validate_input(input_path_obj)

    text_columns = (
        ["content"]
        if text_mode == "content"
        else [
            column
            for column in TEXT_COLUMNS
            if column in available_columns
        ]
    )

    torch, SentenceTransformer = require_dependencies()
    device = choose_device(device_request, torch)
    batch_size = choose_batch_size(device, batch_size_request)

    LOGGER.info("Input: %s", input_path_obj.resolve())
    LOGGER.info("Số chunk: %s", f"{row_count:,}")
    LOGGER.info("Device: %s", device)
    LOGGER.info("Batch size: %s", batch_size)
    LOGGER.info("Các cột tạo text: %s", text_columns)

    model = load_model(
        model_name=model_name,
        device=device,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        max_seq_length=max_seq_length,
        SentenceTransformer=SentenceTransformer,
    )

    full_dimension = detect_full_dimension(model, device)

    if embedding_dim > full_dimension:
        raise ValueError(
            f"Model trả về tối đa {full_dimension} chiều, "
            f"nhưng bạn yêu cầu {embedding_dim}."
        )

    LOGGER.info(
        "Dimension model=%s, dimension lưu=%s",
        full_dimension,
        embedding_dim,
    )

    export_metadata(
        input_path=input_path_obj,
        output_path=metadata_path,
        available_columns=available_columns,
        overwrite=overwrite,
    )

    embeddings, state = prepare_storage(
        embeddings_path=embeddings_path,
        state_path=state_path,
        row_count=row_count,
        embedding_dim=embedding_dim,
        model_name=model_name,
        input_path=input_path_obj,
        overwrite=overwrite,
    )

    config = {
        "input_path": str(input_path_obj.resolve()),
        "model": model_name,
        "device": device,
        "batch_size": batch_size,
        "read_batch_size": read_batch_size,
        "max_seq_length": max_seq_length,
        "full_model_dimension": full_dimension,
        "saved_embedding_dimension": embedding_dim,
        "text_mode": text_mode,
        "text_columns": text_columns,
        "normalize_embeddings": True,
        "row_count": row_count,
        "dtype": "float32",
        "document_prompt": None,
        "recommended_query_prompt_name": "query",
    }
    save_json(config_path, config)

    embed_parquet(
        input_path=input_path_obj,
        model=model,
        embeddings=embeddings,
        state=state,
        state_path=state_path,
        text_columns=text_columns,
        text_mode=text_mode,
        batch_size=batch_size,
        read_batch_size=read_batch_size,
        target_dimension=embedding_dim,
        device=device,
    )

    validation = validate_result(embeddings)
    config["validation"] = validation
    save_json(config_path, config)

    print("\n" + "=" * 72)
    print("QWEN3 DOCUMENT EMBEDDING COMPLETED")
    print("=" * 72)
    print(f"Model       : {model_name}")
    print(f"Rows        : {row_count:,}")
    print(f"Shape       : ({row_count:,}, {embedding_dim})")
    print(f"Embeddings  : {embeddings_path}")
    print(f"Metadata    : {metadata_path}")
    print(f"Config      : {config_path}")
    print(f"Checkpoint  : {state_path}")


# ============================================================
# SỬA THAM SỐ TRỰC TIẾP Ở ĐÂY
# ============================================================

if __name__ == "__main__":
    run_embedding(
        input_path="./data/legal_chunks.parquet",
        output_dir="./embedding_output",
        model_name="Qwen/Qwen3-Embedding-0.6B",
        device_request="cpu",
        batch_size_request=4,
        read_batch_size=64,
        max_seq_length=512,
        embedding_dim=1024,
        text_mode="content",
        cache_dir=None,
        local_files_only=False,
        overwrite=False,
        log_level="INFO",
    )
