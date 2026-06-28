# Thay hàm _parse_chunk_ids trong runner.py bằng hàm này.

import ast
import json
import re
from typing import Any


_QUOTED_TOKEN_PATTERN = re.compile(r"""['"]([^'"]+)['"]""")


@staticmethod
def _parse_chunk_ids(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    if hasattr(value, "tolist"):
        try:
            converted = value.tolist()

            if isinstance(converted, list):
                return [
                    str(item).strip()
                    for item in converted
                    if str(item).strip()
                ]
        except Exception:
            pass

    if isinstance(value, str):
        text = value.strip()

        if not text:
            return []

        try:
            parsed = json.loads(text)

            if isinstance(parsed, (list, tuple, set)):
                return [
                    str(item).strip()
                    for item in parsed
                    if str(item).strip()
                ]

            return [str(parsed).strip()]
        except json.JSONDecodeError:
            pass

        try:
            parsed = ast.literal_eval(text)

            if isinstance(parsed, (list, tuple, set)):
                return [
                    str(item).strip()
                    for item in parsed
                    if str(item).strip()
                ]

            return [str(parsed).strip()]
        except (ValueError, SyntaxError):
            pass

        quoted_tokens = _QUOTED_TOKEN_PATTERN.findall(text)

        if quoted_tokens:
            return [
                token.strip()
                for token in quoted_tokens
                if token.strip()
            ]

        cleaned = text.strip("[](){}")
        tokens = re.split(r"[\s,]+", cleaned)

        return [
            token.strip("'\" ")
            for token in tokens
            if token.strip("'\" ")
        ]

    return [str(value).strip()]
