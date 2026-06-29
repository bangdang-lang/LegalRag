from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from core.config import AppConfig
from graph_rerank.graph_search_pipeline import GraphSearchPipeline
from models.answer_generator import LLMAnswerGenerator
from retrieval.schemas import RetrievalResult
from utils.logger import configure_logging


SUPPORTED_METHODS = tuple(GraphSearchPipeline.SUPPORTED_METHODS)


class LegalRAGDemo:
    """
    Demo hỏi đáp tương tác cho LegalRAG.

    Pipeline:
        1. Nhận query từ người dùng.
        2. Retrieval bằng một trong 5 phương pháp.
        3. Lấy top 5 chunk ở bước xếp hạng cuối cùng.
        4. In rank, score, source, metadata và nội dung chunk.
        5. Dùng đúng top 5 chunk đó để sinh câu trả lời.
    """

    def __init__(
        self,
        config_path: str | Path = "config.json",
        method: str = "hybrid+graph",
        top_k: int = 5,
        show_full_text: bool = True,
        auto_enable_generation: bool = True,
    ) -> None:
        self.config = AppConfig.load(config_path)
        configure_logging(self.config.get("project.log_level", "INFO"))
        self.logger = logging.getLogger(self.__class__.__name__)

        method = method.lower().strip()
        if method not in SUPPORTED_METHODS:
            raise ValueError(
                f"Phương pháp {method!r} không hợp lệ. "
                f"Các phương pháp hỗ trợ: {SUPPORTED_METHODS}"
            )

        if top_k <= 0:
            raise ValueError("top_k phải lớn hơn 0.")

        self.method = method
        self.top_k = top_k
        self.show_full_text = show_full_text

        # config.json của repo đang để generation.enabled=false.
        # Demo cần sinh câu trả lời nên có thể bật trong bộ nhớ lúc chạy.
        if auto_enable_generation and not bool(
            self.config.get("generation.enabled", False)
        ):
            self.config.data.setdefault("generation", {})["enabled"] = True
            self.logger.warning(
                "generation.enabled đang là false; demo đã tạm bật generation "
                "trong bộ nhớ. File config.json không bị thay đổi."
            )

        self.pipeline = GraphSearchPipeline(self.config)
        self.generator = LLMAnswerGenerator(self.config)

        self.chunk_lookup: pd.DataFrame | None = None
        self.id_column = str(self.config.get("embedding.id_column", "id"))
        self.text_column = str(self.config.get("embedding.text_column", "content"))

    def load(self) -> "LegalRAGDemo":
        """Load index retrieval, graph và bảng tra nội dung chunk một lần."""

        self.logger.info("Đang load retrieval indexes và graph...")
        self.pipeline.load()

        chunk_path = self.config.path("legal_chunks")
        if not chunk_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy file chunk: {chunk_path}\n"
                "Hãy build/chunk dữ liệu trước khi chạy demo."
            )

        chunks = pd.read_parquet(chunk_path)

        if self.id_column not in chunks.columns:
            raise KeyError(
                f"File {chunk_path} không có cột ID {self.id_column!r}. "
                f"Các cột hiện có: {list(chunks.columns)}"
            )

        if self.text_column not in chunks.columns:
            raise KeyError(
                f"File {chunk_path} không có cột nội dung {self.text_column!r}. "
                f"Các cột hiện có: {list(chunks.columns)}"
            )

        chunks[self.id_column] = chunks[self.id_column].astype(str).str.strip()
        chunks = chunks.drop_duplicates(subset=[self.id_column], keep="first")
        self.chunk_lookup = chunks.set_index(self.id_column, drop=False)

        self.logger.info(
            "Đã load %d chunks để tra cứu nội dung.",
            len(self.chunk_lookup),
        )
        return self

    @staticmethod
    def _clean_value(value: Any) -> Any:
        """Đổi NaN thành None để in metadata sạch hơn."""

        if value is None:
            return None

        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass

        return value

    def _hydrate_result(self, result: RetrievalResult) -> RetrievalResult:
        """
        Bổ sung text và metadata cho kết quả graph.

        GraphSearchPipeline có thể tạo RetrievalResult chỉ gồm chunk_id,
        score và source. LLM cần nội dung thật nên phải tra lại
        legal_chunks.parquet trước khi generate.
        """

        chunk_id = str(result.chunk_id).strip()

        if self.chunk_lookup is None:
            raise RuntimeError("Demo chưa được load. Hãy gọi load() trước.")

        if chunk_id not in self.chunk_lookup.index:
            self.logger.warning(
                "Không tìm thấy chunk_id=%s trong legal_chunks.parquet.",
                chunk_id,
            )
            return result

        row = self.chunk_lookup.loc[chunk_id]

        # Phòng trường hợp index vẫn trả về DataFrame.
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        row_metadata = {
            str(key): self._clean_value(value)
            for key, value in row.to_dict().items()
        }

        merged_metadata = dict(row_metadata)
        merged_metadata.update(result.metadata or {})

        text = str(result.text or "").strip()
        if not text:
            text = str(row.get(self.text_column, "") or "").strip()

        return RetrievalResult(
            chunk_id=chunk_id,
            score=float(result.score),
            text=text,
            source=str(result.source or ""),
            metadata=merged_metadata,
        )

    def retrieve(self, query: str) -> list[RetrievalResult]:
        """Chạy retrieval và trả về đúng top-k kết quả cuối cùng."""

        query = query.strip()
        if not query:
            raise ValueError("Query không được để trống.")

        raw_results = self.pipeline.search_method(
            query=query,
            method=self.method,
            top_k=self.top_k,
        )

        return [
            self._hydrate_result(result)
            for result in raw_results
        ]

    def _display_text(self, text: str) -> str:
        if self.show_full_text:
            return text

        max_chars = 1200
        if len(text) <= max_chars:
            return text

        return text[:max_chars].rstrip() + "..."

    def print_results(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> None:
        """In top-k chunk với rank và điểm của bước cuối cùng."""

        print("\n" + "=" * 110)
        print("KẾT QUẢ RETRIEVAL CUỐI CÙNG")
        print("=" * 110)
        print(f"Query       : {query}")
        print(f"Method      : {self.method}")
        print(f"Top-k       : {len(results)}")
        print("=" * 110)

        if not results:
            print("Không tìm thấy chunk phù hợp.")
            return

        for rank, item in enumerate(results, start=1):
            metadata = item.metadata or {}

            document_number = self._clean_value(
                metadata.get("document_number")
            )
            title = self._clean_value(metadata.get("title"))
            article = self._clean_value(
                metadata.get("articles", metadata.get("article"))
            )
            url = self._clean_value(metadata.get("url"))

            print(f"\n{'-' * 110}")
            print(f"RANK       : {rank}")
            print(f"SCORE      : {float(item.score):.8f}")
            print(f"SOURCE     : {item.source or 'unknown'}")
            print(f"CHUNK ID   : {item.chunk_id}")

            if document_number:
                print(f"VĂN BẢN    : {document_number}")
            if title:
                print(f"TIÊU ĐỀ    : {title}")
            if article:
                print(f"ĐIỀU/MỤC   : {article}")
            if url:
                print(f"URL        : {url}")

            print("NỘI DUNG:")
            print(self._display_text(item.text) or "[Chunk không có nội dung]")

        print("\n" + "=" * 110)

    def generate_answer(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> str:
        """Sinh câu trả lời chỉ từ các chunk cuối cùng vừa được in."""

        usable_results = [
            item for item in results if str(item.text).strip()
        ]

        if not usable_results:
            return (
                "Không thể sinh câu trả lời vì các kết quả retrieval "
                "không có nội dung chunk."
            )

        return self.generator.generate(query, usable_results)

    def ask(self, query: str) -> str:
        """Retrieval, in top chunk rồi sinh và in câu trả lời."""

        results = self.retrieve(query)
        self.print_results(query, results)

        print("\nĐang sinh câu trả lời từ các chunk phía trên...\n")
        answer = self.generate_answer(query, results)

        print("=" * 110)
        print("CÂU TRẢ LỜI")
        print("=" * 110)
        print(answer)
        print("=" * 110)

        return answer

    def interactive(self) -> None:
        """Chạy vòng lặp nhập query từ terminal."""

        print("\n" + "=" * 110)
        print("LEGAL RAG INTERACTIVE DEMO")
        print("=" * 110)
        print(f"Retrieval method : {self.method}")
        print(f"Final top-k      : {self.top_k}")
        print("Nhập 'exit', 'quit' hoặc 'q' để thoát.")
        print("=" * 110)

        while True:
            try:
                query = input("\nNhập câu hỏi pháp luật: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nĐã thoát demo.")
                break

            if query.casefold() in {"exit", "quit", "q"}:
                print("Đã thoát demo.")
                break

            if not query:
                print("Query không được để trống.")
                continue

            try:
                self.ask(query)
            except Exception as error:
                self.logger.exception("Không thể xử lý query.")
                print(f"\nLỗi: {error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive LegalRAG demo: nhập query, retrieval top chunks "
            "và sinh câu trả lời."
        )
    )

    parser.add_argument(
        "--config",
        default="config.json",
        help="Đường dẫn config.json.",
    )
    parser.add_argument(
        "--method",
        choices=SUPPORTED_METHODS,
        default="hybrid+graph",
        help="Phương pháp retrieval.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Số chunk cuối cùng dùng để in và generate answer.",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Chạy một query rồi thoát. Bỏ qua để chạy chế độ tương tác.",
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Rút gọn nội dung chunk khi in ra terminal.",
    )
    parser.add_argument(
        "--no-auto-enable-generation",
        action="store_true",
        help=(
            "Không tự bật generation trong bộ nhớ. Khi dùng tùy chọn này, "
            "generation.enabled trong config.json phải là true."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        demo = LegalRAGDemo(
            config_path=args.config,
            method=args.method,
            top_k=args.top_k,
            show_full_text=not args.truncate,
            auto_enable_generation=not args.no_auto_enable_generation,
        ).load()

        if args.query:
            demo.ask(args.query)
        else:
            demo.interactive()

    except Exception as error:
        logging.getLogger("LegalRAGDemo").exception(
            "Demo khởi động thất bại."
        )
        print(f"\nKhông thể chạy demo: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
