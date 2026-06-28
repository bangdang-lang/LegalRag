from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMAnswerGenerator:
    """
    Sinh câu trả lời từ các kết quả retrieval/graph rerank.

    Đầu vào:
        - query của người dùng
        - danh sách kết quả retrieval hoặc graph rerank

    Mỗi phần tử kết quả có thể là:
        - object có các thuộc tính: chunk_id, text, metadata, final_score
        - dict có các khóa tương ứng

    Module yêu cầu model sinh văn bản kiểu Causal Language Model,
    ví dụ Qwen2.5-Instruct hoặc Qwen3-Instruct.
    """

    TEXT_KEYS = (
        "text",
        "chunk_text",
        "content",
        "chunk",
        "page_content",
        "article_text",
    )

    TITLE_KEYS = (
        "title",
        "document_title",
        "name",
        "document_number",
    )

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        device: str | None = None,
        max_context_chars: int = 18000,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.9,
        do_sample: bool = False,
        local_files_only: bool = True,
    ) -> None:
        self.model_name_or_path = str(model_name_or_path)
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.max_context_chars = max_context_chars
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.local_files_only = local_files_only

        self.tokenizer: Any | None = None
        self.model: Any | None = None

    def load(self) -> "LLMAnswerGenerator":
        model_source = Path(self.model_name_or_path)

        if self.local_files_only:
            model_source = model_source.resolve()

            if not model_source.exists():
                raise FileNotFoundError(
                    "Không tìm thấy thư mục LLM local:\n"
                    f"{model_source}"
                )

            if not (model_source / "config.json").exists():
                raise FileNotFoundError(
                    "Thư mục model tồn tại nhưng thiếu config.json:\n"
                    f"{model_source}"
                )

        source = str(model_source)

        print(f"Loading LLM from: {source}")
        print(f"LLM device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            source,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )

        dtype = (
            torch.float16
            if self.device == "cuda"
            else torch.float32
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            source,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

        self.model.to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        return self

    @staticmethod
    def _get_value(
        item: Any,
        key: str,
        default: Any = None,
    ) -> Any:
        if isinstance(item, dict):
            return item.get(key, default)

        return getattr(item, key, default)

    def _extract_text(self, item: Any) -> str:
        direct_text = self._get_value(item, "text", "")
        if direct_text:
            return str(direct_text).strip()

        metadata = self._get_value(item, "metadata", {}) or {}

        if isinstance(metadata, dict):
            for key in self.TEXT_KEYS:
                value = metadata.get(key)
                if value:
                    return str(value).strip()

        return ""

    def _extract_title(self, item: Any) -> str:
        metadata = self._get_value(item, "metadata", {}) or {}

        if isinstance(metadata, dict):
            for key in self.TITLE_KEYS:
                value = metadata.get(key)
                if value:
                    return str(value).strip()

        return ""

    def build_context(
        self,
        results: Iterable[Any],
    ) -> str:
        """
        Chuyển danh sách kết quả retrieval thành context có đánh số nguồn.
        """
        context_parts: list[str] = []
        current_length = 0

        for rank, item in enumerate(results, start=1):
            chunk_id = str(
                self._get_value(item, "chunk_id", f"chunk_{rank}")
            )

            text = self._extract_text(item)
            if not text:
                continue

            title = self._extract_title(item)

            final_score = self._get_value(item, "final_score")
            if final_score is None:
                final_score = self._get_value(item, "rrf_score")

            header = f"[Nguồn {rank}] chunk_id={chunk_id}"

            if title:
                header += f" | văn bản={title}"

            if final_score is not None:
                try:
                    header += f" | score={float(final_score):.6f}"
                except (TypeError, ValueError):
                    pass

            block = f"{header}\n{text.strip()}\n"

            if current_length + len(block) > self.max_context_chars:
                remaining = self.max_context_chars - current_length

                if remaining > 300:
                    context_parts.append(block[:remaining])

                break

            context_parts.append(block)
            current_length += len(block)

        if not context_parts:
            return "Không có ngữ cảnh pháp luật phù hợp được truy xuất."

        return "\n".join(context_parts)

    @staticmethod
    def build_messages(
        query: str,
        context: str,
    ) -> list[dict[str, str]]:
        system_prompt = """
Bạn là trợ lý hỏi đáp pháp luật Việt Nam.

Quy tắc bắt buộc:
1. Chỉ trả lời dựa trên các nguồn được cung cấp trong phần NGỮ CẢNH.
2. Không tự bịa điều luật, số văn bản, cơ quan ban hành hoặc nội dung pháp lý.
3. Nếu ngữ cảnh không đủ để trả lời, phải nói rõ rằng chưa đủ thông tin.
4. Khi sử dụng thông tin từ một đoạn, trích dẫn bằng dạng [Nguồn 1], [Nguồn 2].
5. Ưu tiên câu trả lời trực tiếp, rõ ràng và có cấu trúc.
6. Nếu các nguồn mâu thuẫn, phải nêu rõ sự mâu thuẫn.
7. Không khẳng định đây là tư vấn pháp lý chính thức.
""".strip()

        user_prompt = f"""
CÂU HỎI:
{query.strip()}

NGỮ CẢNH:
{context}

Hãy trả lời câu hỏi dựa hoàn toàn trên ngữ cảnh trên.
Cuối câu trả lời, thêm mục "Nguồn sử dụng" liệt kê các nguồn thực sự đã dùng.
""".strip()

        return [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

    @torch.inference_mode()
    def generate_answer(
        self,
        query: str,
        results: Iterable[Any],
    ) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "LLM chưa được load. Hãy gọi generator.load() trước."
            )

        query = query.strip()
        if not query:
            raise ValueError("Query không được để trống.")

        context = self.build_context(results)
        messages = self.build_messages(query, context)

        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = (
                f"SYSTEM:\n{messages[0]['content']}\n\n"
                f"USER:\n{messages[1]['content']}\n\n"
                "ASSISTANT:\n"
            )

        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
        )

        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if self.do_sample:
            generation_kwargs["temperature"] = self.temperature
            generation_kwargs["top_p"] = self.top_p

        output_ids = self.model.generate(
            **encoded,
            **generation_kwargs,
        )

        generated_ids = output_ids[
            0,
            encoded["input_ids"].shape[1]:,
        ]

        answer = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

        if not answer:
            return (
                "Không thể sinh câu trả lời từ ngữ cảnh hiện tại. "
                "Vui lòng kiểm tra model hoặc dữ liệu retrieval."
            )

        return answer


def answer_from_pipeline(
    query: str,
    results: Iterable[Any],
    *,
    model_path: str | Path,
) -> str:
    """
    Hàm gọi nhanh khi không cần giữ model trong bộ nhớ qua nhiều câu hỏi.
    """
    generator = LLMAnswerGenerator(
        model_name_or_path=model_path,
        local_files_only=True,
    ).load()

    return generator.generate_answer(
        query=query,
        results=results,
    )
