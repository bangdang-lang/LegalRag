from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from .rrf import reciprocal_rank_fusion
from .schemas import RetrievalResult


class HybridRetriever:
    """
    Pipeline retrieval:
        query -> query embedding -> BM25 search + FAISS search
              -> Reciprocal Rank Fusion -> top-k

    Module được thiết kế theo cấu trúc thư mục trong project:
        bm25_output/
        embedding_output/
        faiss_output/
        models/
        src/retrieval/
    """

    TEXT_COLUMN_CANDIDATES = (
        "text",
        "chunk_text",
        "content",
        "chunk",
        "page_content",
        "article_text",
    )
    ID_COLUMN_CANDIDATES = (
        "chunk_id",
        "id",
        "node_id",
        "document_chunk_id",
    )

    def __init__(
        self,
        *,
        project_root: str | Path = ".",
        model_name_or_path: str | Path = "Qwen/Qwen3-Embedding-0.6B",
        bm25_dir: str | Path = "bm25_output",
        faiss_index_path: str | Path = "faiss_output/legal_chunks.faiss",
        bm25_lookup_path: str | Path = "bm25_output/bm25_lookup.parquet",
        embedding_metadata_path: str | Path = "embedding_output/embedding_metadata.parquet",
        device: str | None = None,
        max_length: int = 512,
        query_instruction: str = (
            "Given a Vietnamese legal question, retrieve relevant legal passages."
        ),
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.model_path = self._resolve_path(model_name_or_path)
        self.bm25_dir = self._resolve_path(bm25_dir)
        self.faiss_index_path = self._resolve_path(faiss_index_path)
        self.bm25_lookup_path = self._resolve_path(bm25_lookup_path)
        self.embedding_metadata_path = self._resolve_path(embedding_metadata_path)

        self.max_length = max_length
        self.query_instruction = query_instruction
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer: Any | None = None
        self.embedding_model: Any | None = None
        self.bm25_retriever: Any | None = None
        self.faiss_index: Any | None = None

        self.bm25_lookup: pd.DataFrame | None = None
        self.embedding_metadata: pd.DataFrame | None = None

        self.bm25_id_column: str | None = None
        self.bm25_text_column: str | None = None
        self.faiss_id_column: str | None = None
        self.faiss_text_column: str | None = None

        self._bm25_row_by_id: dict[str, dict[str, Any]] = {}
        self._faiss_row_by_id: dict[str, dict[str, Any]] = {}

    def _resolve_path(self, path: str | Path) -> Path:
        value = Path(path)
        return value if value.is_absolute() else self.project_root / value

    @staticmethod
    def _find_column(
        dataframe: pd.DataFrame,
        candidates: tuple[str, ...],
        *,
        required: bool = True,
    ) -> str | None:
        columns_lower = {str(column).lower(): str(column) for column in dataframe.columns}

        for candidate in candidates:
            if candidate.lower() in columns_lower:
                return columns_lower[candidate.lower()]

        if required:
            raise ValueError(
                f"Không tìm thấy cột phù hợp. Các cột hiện có: "
                f"{list(dataframe.columns)}; cần một trong: {list(candidates)}"
            )
        return None

    @staticmethod
    def _row_to_dict(row: pd.Series) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in row.to_dict().items():
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, float) and np.isnan(value):
                value = None
            output[str(key)] = value
        return output

    def load(self) -> "HybridRetriever":
        self._validate_paths()
        self._load_embedding_model()
        self._load_bm25()
        self._load_faiss()
        self._load_lookup_tables()
        return self

    def _validate_paths(self) -> None:
        required_paths = [
            self.bm25_dir,
            self.faiss_index_path,
            self.bm25_lookup_path,
            self.embedding_metadata_path,
        ]

        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Không tìm thấy các đường dẫn sau:\n- " + "\n- ".join(missing)
            )

    def _load_embedding_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
        )
        self.embedding_model = AutoModel.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        self.embedding_model.to(self.device)
        self.embedding_model.eval()

    def _load_bm25(self) -> None:
        try:
            import bm25s
        except ImportError as exc:
            raise ImportError(
                "Thiếu thư viện bm25s. Cài bằng: pip install bm25s"
            ) from exc

        # Thư mục bm25_output trong ảnh là index đã được bm25s.save().
        self.bm25_retriever = bm25s.BM25.load(
            str(self.bm25_dir),
            load_corpus=False,
        )

    def _load_faiss(self) -> None:
        self.faiss_index = faiss.read_index(str(self.faiss_index_path))

    def _load_lookup_tables(self) -> None:
        self.bm25_lookup = pd.read_parquet(self.bm25_lookup_path)
        self.embedding_metadata = pd.read_parquet(self.embedding_metadata_path)

        self.bm25_id_column = self._find_column(
            self.bm25_lookup,
            self.ID_COLUMN_CANDIDATES,
        )
        self.bm25_text_column = self._find_column(
            self.bm25_lookup,
            self.TEXT_COLUMN_CANDIDATES,
        )

        self.faiss_id_column = self._find_column(
            self.embedding_metadata,
            self.ID_COLUMN_CANDIDATES,
        )
        self.faiss_text_column = self._find_column(
            self.embedding_metadata,
            self.TEXT_COLUMN_CANDIDATES,
            required=False,
        )

        self._bm25_row_by_id = {
            str(row[self.bm25_id_column]): self._row_to_dict(row)
            for _, row in self.bm25_lookup.iterrows()
        }
        self._faiss_row_by_id = {
            str(row[self.faiss_id_column]): self._row_to_dict(row)
            for _, row in self.embedding_metadata.iterrows()
        }

        if len(self.embedding_metadata) != self.faiss_index.ntotal:
            raise ValueError(
                "Số dòng embedding_metadata không bằng số vector FAISS: "
                f"{len(self.embedding_metadata)} != {self.faiss_index.ntotal}. "
                "Metadata phải giữ đúng thứ tự lúc tạo index."
            )

    @staticmethod
    def _last_token_pool(
        last_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pooling theo token cuối có hiệu lực, phù hợp với Qwen3-Embedding.
        Hoạt động với cả left padding và right padding.
        """
        left_padding = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]))

        if left_padding:
            return last_hidden_states[:, -1]

        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths,
        ]

    def _format_query(self, query: str) -> str:
        query = query.strip()
        if not query:
            raise ValueError("Query không được để trống.")

        return (
            f"Instruct: {self.query_instruction}\n"
            f"Query: {query}"
        )

    @torch.inference_mode()
    def embed_query(self, query: str) -> np.ndarray:
        if self.tokenizer is None or self.embedding_model is None:
            raise RuntimeError("Chưa load model. Hãy gọi retriever.load() trước.")

        encoded = self.tokenizer(
            [self._format_query(query)],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        outputs = self.embedding_model(**encoded)
        embedding = self._last_token_pool(
            outputs.last_hidden_state,
            encoded["attention_mask"],
        )
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

        return embedding.detach().cpu().numpy().astype("float32")

    def search_bm25(self, query: str, top_k: int = 30) -> list[dict[str, Any]]:
        if self.bm25_retriever is None or self.bm25_lookup is None:
            raise RuntimeError("Chưa load BM25. Hãy gọi retriever.load() trước.")

        try:
            import bm25s
        except ImportError as exc:
            raise ImportError("Thiếu bm25s.") from exc

        query_tokens = bm25s.tokenize(
            [query],
            stopwords=None,
            stemmer=None,
            show_progress=False,
        )

        document_indices, scores = self.bm25_retriever.retrieve(
            query_tokens,
            k=min(top_k, len(self.bm25_lookup)),
        )

        results: list[dict[str, Any]] = []
        for rank, (row_index, score) in enumerate(
            zip(document_indices[0], scores[0]),
            start=1,
        ):
            row = self.bm25_lookup.iloc[int(row_index)]
            row_dict = self._row_to_dict(row)
            results.append(
                {
                    "chunk_id": str(row[self.bm25_id_column]),
                    "text": str(row[self.bm25_text_column]),
                    "rank": rank,
                    "score": float(score),
                    "metadata": row_dict,
                }
            )

        return results

    def search_faiss(self, query: str, top_k: int = 30) -> list[dict[str, Any]]:
        if self.faiss_index is None or self.embedding_metadata is None:
            raise RuntimeError("Chưa load FAISS. Hãy gọi retriever.load() trước.")

        query_vector = self.embed_query(query)
        k = min(top_k, self.faiss_index.ntotal)
        scores, indices = self.faiss_index.search(query_vector, k)

        results: list[dict[str, Any]] = []
        for rank, (row_index, score) in enumerate(
            zip(indices[0], scores[0]),
            start=1,
        ):
            if int(row_index) < 0:
                continue

            row = self.embedding_metadata.iloc[int(row_index)]
            chunk_id = str(row[self.faiss_id_column])
            row_dict = self._row_to_dict(row)

            text = ""
            if self.faiss_text_column is not None:
                text = str(row[self.faiss_text_column])
            elif chunk_id in self._bm25_row_by_id:
                text = str(
                    self._bm25_row_by_id[chunk_id].get(self.bm25_text_column, "")
                )

            results.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "rank": rank,
                    "score": float(score),
                    "metadata": row_dict,
                }
            )

        return results

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        bm25_top_k: int = 30,
        faiss_top_k: int = 30,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        faiss_weight: float = 1.0,
    ) -> list[RetrievalResult]:
        """
        Chạy retrieval song song về mặt logic, sau đó hợp nhất bằng RRF.
        """
        bm25_results = self.search_bm25(query, top_k=bm25_top_k)
        faiss_results = self.search_faiss(query, top_k=faiss_top_k)

        rrf_scores = reciprocal_rank_fusion(
            [bm25_results, faiss_results],
            id_key="chunk_id",
            k=rrf_k,
            weights=[bm25_weight, faiss_weight],
        )

        bm25_by_id = {item["chunk_id"]: item for item in bm25_results}
        faiss_by_id = {item["chunk_id"]: item for item in faiss_results}

        fused_results: list[RetrievalResult] = []

        for chunk_id, rrf_score in rrf_scores.items():
            bm25_item = bm25_by_id.get(chunk_id)
            faiss_item = faiss_by_id.get(chunk_id)

            source_item = bm25_item or faiss_item
            text = str(source_item.get("text", "")) if source_item else ""

            metadata: dict[str, Any] = {}
            if faiss_item:
                metadata.update(faiss_item.get("metadata", {}))
            if bm25_item:
                metadata.update(bm25_item.get("metadata", {}))

            fused_results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    text=text,
                    rrf_score=float(rrf_score),
                    bm25_rank=bm25_item["rank"] if bm25_item else None,
                    bm25_score=bm25_item["score"] if bm25_item else None,
                    faiss_rank=faiss_item["rank"] if faiss_item else None,
                    faiss_score=faiss_item["score"] if faiss_item else None,
                    metadata=metadata,
                )
            )

        fused_results.sort(
            key=lambda item: (
                -item.rrf_score,
                item.bm25_rank or 10**9,
                item.faiss_rank or 10**9,
                item.chunk_id,
            )
        )

        return fused_results[:top_k]

    def print_results(self, results: list[RetrievalResult]) -> None:
        for position, result in enumerate(results, start=1):
            print("=" * 100)
            print(f"TOP {position}")
            print(f"chunk_id   : {result.chunk_id}")
            print(f"rrf_score  : {result.rrf_score:.8f}")
            print(f"bm25_rank  : {result.bm25_rank}")
            print(f"bm25_score : {result.bm25_score}")
            print(f"faiss_rank : {result.faiss_rank}")
            print(f"faiss_score: {result.faiss_score}")
            print("-" * 100)
            print(result.text)
