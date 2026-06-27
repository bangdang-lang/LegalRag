from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

try:
    from huggingface_hub import snapshot_download
except ImportError as exc:
    raise ImportError(
        "Chưa cài huggingface_hub. Chạy: "
        "python -m pip install --upgrade huggingface_hub"
    ) from exc


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

LOGGER = logging.getLogger(__name__)

ModelKind = Literal["embedding", "llm"]


@dataclass(frozen=True)
class ModelSpec:
    """Cấu hình của một model chỉ dùng cho bước tải xuống."""

    kind: ModelKind
    repo_id: str
    local_dir: Path
    revision: str = "main"


def _resolve_path(path_value: str | Path, project_dir: Path) -> Path:
    """Chuyển đường dẫn tương đối thành đường dẫn nằm trong project hiện tại."""

    path = Path(path_value).expanduser()

    if not path.is_absolute():
        path = project_dir / path

    return path.resolve()


def _read_json(path: Path) -> dict[str, Any]:
    """Đọc file JSON và kiểm tra object gốc."""

    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file config: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("Nội dung config phải là một JSON object.")

    return data


def _get_model_section(
    config: dict[str, Any],
    kind: ModelKind,
) -> dict[str, Any]:
    """
    Hỗ trợ cả config mới lẫn cấu trúc config cũ của bạn.

    Dạng mới:
        "models": {
            "embedding": {...},
            "llm": {...}
        }

    Dạng cũ:
        "embedding_model": {
            "name": "...",
            "cache": "..."
        },
        "llm_model": {
            "name": "...",
            "cache": "..."
        }
    """

    models = config.get("models")

    if isinstance(models, dict):
        section = models.get(kind)

        if isinstance(section, dict):
            return section

    legacy_key = (
        "embedding_model"
        if kind == "embedding"
        else "llm_model"
    )

    section = config.get(legacy_key)

    if isinstance(section, dict):
        return section

    raise KeyError(
        f"Không tìm thấy cấu hình model '{kind}'. "
        f"Hãy thêm models.{kind} hoặc {legacy_key} vào config."
    )


def _build_model_spec(
    config: dict[str, Any],
    kind: ModelKind,
    project_dir: Path,
) -> ModelSpec:
    """Chuyển cấu hình JSON thành ModelSpec."""

    section = _get_model_section(config, kind)

    repo_id = section.get("repo_id") or section.get("name")
    local_dir = section.get("local_dir") or section.get("cache")
    revision = str(section.get("revision", "main")).strip()

    if not repo_id or not isinstance(repo_id, str):
        raise ValueError(
            f"Model '{kind}' thiếu repo_id/name hợp lệ."
        )

    if not local_dir or not isinstance(local_dir, str):
        raise ValueError(
            f"Model '{kind}' thiếu local_dir/cache hợp lệ."
        )

    return ModelSpec(
        kind=kind,
        repo_id=repo_id.strip(),
        local_dir=_resolve_path(local_dir, project_dir),
        revision=revision or "main",
    )


def _get_hf_settings(
    config: dict[str, Any],
    project_dir: Path,
) -> tuple[Path, str | None]:
    """Lấy cache directory và token từ biến môi trường."""

    hf_config = config.get("huggingface", {})

    if not isinstance(hf_config, dict):
        hf_config = {}

    cache_value = hf_config.get(
        "cache_dir",
        "./.hf_cache",
    )

    token_env = str(
        hf_config.get(
            "token_env",
            "HF_TOKEN",
        )
    )

    cache_dir = _resolve_path(
        cache_value,
        project_dir,
    )

    token = os.getenv(token_env)

    return cache_dir, token


def _write_manifest(
    spec: ModelSpec,
    snapshot_path: Path,
) -> None:
    """Ghi thông tin tải xuống mà không mở hoặc load trọng số."""

    manifest = {
        "kind": spec.kind,
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "local_dir": str(spec.local_dir),
        "snapshot_path": str(snapshot_path),
        "loaded_into_ram": False,
    }

    manifest_path = (
        spec.local_dir
        / "download_manifest.json"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            ensure_ascii=False,
            indent=2,
        )


def download_model(
    spec: ModelSpec,
    cache_dir: Path,
    token: str | None = None,
) -> Path:
    """
    Tải toàn bộ repository model về ổ đĩa.

    Hàm này KHÔNG:
    - gọi AutoModel.from_pretrained();
    - gọi AutoTokenizer.from_pretrained();
    - gọi SentenceTransformer();
    - đọc tensor trọng số vào RAM.
    """

    spec.local_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOGGER.info(
        "Đang tải %s model: %s",
        spec.kind,
        spec.repo_id,
    )
    LOGGER.info(
        "Thư mục đích: %s",
        spec.local_dir,
    )

    downloaded_path = snapshot_download(
        repo_id=spec.repo_id,
        revision=spec.revision,
        repo_type="model",
        local_dir=str(spec.local_dir),
        cache_dir=str(cache_dir),
        token=token,
    )

    snapshot_path = Path(downloaded_path).resolve()

    _write_manifest(
        spec,
        snapshot_path,
    )

    LOGGER.info(
        "Đã tải xong %s model.",
        spec.kind,
    )

    return snapshot_path


def download_qwen_models(
    config_path: str | Path = "config.json",
    only: Literal["all", "embedding", "llm"] = "all",
) -> dict[str, Path]:
    """
    Tải Qwen embedding và/hoặc Qwen LLM theo config.

    Hàm trả về đường dẫn local để code khác sử dụng về sau.
    Việc gọi hàm này không load model vào RAM.
    """

    config_file = Path(
        config_path
    ).expanduser().resolve()

    project_dir = config_file.parent
    config = _read_json(config_file)

    cache_dir, token = _get_hf_settings(
        config,
        project_dir,
    )

    selected_kinds: list[ModelKind]

    if only == "embedding":
        selected_kinds = ["embedding"]
    elif only == "llm":
        selected_kinds = ["llm"]
    elif only == "all":
        selected_kinds = ["embedding", "llm"]
    else:
        raise ValueError(
            "only phải là all, embedding hoặc llm."
        )

    result: dict[str, Path] = {}

    for kind in selected_kinds:
        spec = _build_model_spec(
            config,
            kind,
            project_dir,
        )

        result[kind] = download_model(
            spec=spec,
            cache_dir=cache_dir,
            token=token,
        )

    return result


def models_are_downloaded(
    config_path: str | Path = "config.json",
) -> dict[str, bool]:
    """Kiểm tra file manifest, không load model."""

    config_file = Path(
        config_path
    ).expanduser().resolve()

    project_dir = config_file.parent
    config = _read_json(config_file)

    status: dict[str, bool] = {}

    for kind in ("embedding", "llm"):
        spec = _build_model_spec(
            config,
            kind,
            project_dir,
        )

        status[kind] = (
            spec.local_dir
            / "download_manifest.json"
        ).exists()

    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tải Qwen embedding và Qwen LLM về thư mục "
            "hiện tại mà không load model vào RAM."
        )
    )

    parser.add_argument(
        "--config",
        default="config.json",
        help="Đường dẫn config JSON.",
    )

    parser.add_argument(
        "--only",
        choices=["all", "embedding", "llm"],
        default="all",
        help="Chọn model cần tải.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    paths = download_qwen_models(
        config_path=args.config,
        only=args.only,
    )

    print("\nĐường dẫn model đã tải:")

    for kind, path in paths.items():
        print(f"- {kind}: {path}")


if __name__ == "__main__":
    main()
