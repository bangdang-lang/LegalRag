from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from core.config import AppConfig


class QwenEmbeddingModel:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.settings = config.section("embedding")
        self.logger = logging.getLogger(self.__class__.__name__)
        self.tokenizer = None
        self.model = None
        self.device = "cpu"

    def load(self) -> "QwenEmbeddingModel":
        import torch
        from transformers import AutoModel, AutoTokenizer
        model_path = self.config.root / self.settings["local_model_dir"]
        source = str(model_path if model_path.exists() else self.settings["model_name"])
        requested = self.settings.get("device", "auto")
        self.device = "cuda" if requested == "auto" and torch.cuda.is_available() else requested
        if self.device == "auto":
            self.device = "cpu"
        dtype = torch.float16 if self.settings.get("dtype") == "float16" and self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(source, trust_remote_code=True, torch_dtype=dtype).to(self.device).eval()
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        import torch
        import torch.nn.functional as F
        if self.model is None or self.tokenizer is None:
            self.load()
        encoded = self.tokenizer(texts, padding=True, truncation=True, max_length=int(self.settings["max_length"]), return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = self.model(**encoded)
            hidden = output.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            vectors = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            if self.settings.get("normalize", True):
                vectors = F.normalize(vectors, p=2, dim=1)
        return vectors.float().cpu().numpy()


class EmbeddingBuilder:
    def __init__(self, config: AppConfig, model: QwenEmbeddingModel | None = None) -> None:
        self.config = config
        self.model = model or QwenEmbeddingModel(config)
        self.logger = logging.getLogger(self.__class__.__name__)

    def build(self) -> tuple[Path, Path]:
        source = self.config.path("legal_chunks")
        output = self.config.path("embeddings")
        metadata_output = self.config.path("embedding_metadata")
        settings = self.config.section("embedding")
        frame = pd.read_parquet(source)
        text_col, id_col = settings["text_column"], settings["id_column"]
        if text_col not in frame or id_col not in frame:
            raise ValueError(f"Input must contain '{id_col}' and '{text_col}'")
        vectors: list[np.ndarray] = []
        batch_size = int(settings["batch_size"])
        texts = frame[text_col].fillna("").astype(str).tolist()
        for start in range(0, len(texts), batch_size):
            vectors.append(self.model.encode(texts[start:start + batch_size]))
            self.logger.info("Embedded %d/%d", min(start + batch_size, len(texts)), len(texts))
        matrix = np.vstack(vectors).astype("float32")
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, matrix)
        metadata_columns = [c for c in frame.columns if c != "embedding"]
        frame[metadata_columns].to_parquet(metadata_output, index=False)
        return output, metadata_output
