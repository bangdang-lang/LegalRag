from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import AppConfig


class FaissIndexBuilder:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def build(self) -> Path:
        import faiss
        vectors = np.load(self.config.path("embeddings")).astype("float32")
        metric = self.config.get("faiss.metric", "ip")
        index = faiss.IndexFlatIP(vectors.shape[1]) if metric == "ip" else faiss.IndexFlatL2(vectors.shape[1])
        index.add(vectors)
        output = self.config.path("faiss_index")
        output.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(output))
        self.logger.info("Saved FAISS index with %d vectors", index.ntotal)
        return output


class BM25IndexBuilder:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def build(self) -> Path:
        import bm25s
        settings = self.config.section("bm25")
        frame = pd.read_parquet(self.config.path("legal_chunks"))
        texts = frame[settings["text_column"]].fillna("").astype(str).tolist()
        corpus_tokens = bm25s.tokenize(texts, stopwords=None)
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        output_dir = self.config.path("bm25_dir")
        output_dir.mkdir(parents=True, exist_ok=True)
        retriever.save(str(output_dir))
        frame.to_parquet(self.config.path("bm25_lookup"), index=False)
        self.logger.info("Saved BM25 index with %d documents", len(frame))
        return output_dir
