"""Làm sạch và tách từ tiếng Việt."""

from __future__ import annotations

import html
import re
import unicodedata
from pathlib import Path

import pandas as pd
from underthesea import word_tokenize


URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: object) -> str:
    """Chuẩn hóa nhẹ văn bản, giữ dấu tiếng Việt, phủ định và dấu câu."""
    if text is None or (not isinstance(text, (list, tuple, dict)) and pd.isna(text)):
        return ""

    value = str(text)
    # NFC đưa các ký tự Unicode tương đương về cùng một cách biểu diễn.
    value = unicodedata.normalize("NFC", value)
    # Chuyển HTML entity (ví dụ &amp;) trước khi bỏ thẻ HTML.
    value = html.unescape(value)
    value = HTML_TAG_PATTERN.sub(" ", value)
    # URL thường không mang nội dung ngôn ngữ hữu ích cho mốc TF-IDF ban đầu.
    value = URL_PATTERN.sub(" ", value)
    value = value.lower()
    # Gom khoảng trắng, tab và xuống dòng; không xóa dấu câu hay emoji.
    value = WHITESPACE_PATTERN.sub(" ", value).strip()
    return value


def segment_vietnamese(text: object) -> str:
    """Tách từ ghép tiếng Việt bằng Underthesea, trả về chuỗi dùng dấu gạch dưới."""
    cleaned = clean_text(text)
    if not cleaned:
        return ""
    return word_tokenize(cleaned, format="text")


def preprocess_dataframe(dataframe: pd.DataFrame, text_column: str) -> pd.DataFrame:
    result = dataframe.copy()
    result["text_original"] = result[text_column].astype(str)
    result["text_clean"] = result[text_column].map(clean_text)
    result["text_segmented"] = result["text_clean"].map(segment_vietnamese)
    return result


def save_preprocessing_examples(dataframe: pd.DataFrame, output_file: Path, count: int = 8) -> None:
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    examples: list[str] = ["VÍ DỤ TIỀN XỬ LÝ VĂN BẢN", "=" * 40]
    for number, (_, row) in enumerate(dataframe.head(count).iterrows(), start=1):
        examples.extend(
            [
                f"\nVí dụ {number}",
                f"Văn bản gốc:\n{row['text_original']}",
                f"\nSau làm sạch:\n{row['text_clean']}",
                f"\nSau tách từ:\n{row['text_segmented']}",
            ]
        )
    output_file.write_text("\n".join(examples) + "\n", encoding="utf-8")
