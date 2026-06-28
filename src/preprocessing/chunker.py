from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class LegalChunk:
    chunk_id: str
    document_id: str
    chunk_type: str
    content: str


class LegalTextChunker:
    ARTICLE_PATTERN = re.compile(r"(?im)^\s*Điều\s+(\d+[a-zA-Z]?)\s*[.:]?\s*")

    def split_articles(self, document_id: str, text: str) -> list[LegalChunk]:
        matches = list(self.ARTICLE_PATTERN.finditer(text))
        chunks: list[LegalChunk] = []
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            content = text[match.start():end].strip()
            if content:
                article = match.group(1)
                chunks.append(LegalChunk(f"{document_id}_dieu_{article}_{index + 1:04d}", document_id, "dieu", content))
        return chunks
