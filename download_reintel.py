"""Tải split train có nhãn của ReINTEL từ Hugging Face."""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download


DATASET_ID = "ReliableAI/ReINTEL"
DATASET_SPLIT = "train"
ORIGINAL_TEXT_COLUMN = "post_message"
ORIGINAL_LABEL_COLUMN = "label"
PROJECT_TEXT_COLUMN = "text"
PROJECT_LABEL_COLUMN = "label"
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = PROJECT_DIR / "data" / "raw" / "reintel_train.csv"
METADATA_FILE = PROJECT_DIR / "data" / "raw" / "reintel_metadata.json"


def verify_label_mapping_from_card(card_text: str) -> dict[int, str]:
    """Chỉ chấp nhận ánh xạ nhãn được ghi rõ trên dataset card chính thức."""
    normalized = re.sub(r"[`*]", "", card_text.lower())
    if "1: unreliable" not in normalized or "0: reliable" not in normalized:
        raise RuntimeError(
            "Không xác minh được ý nghĩa nhãn từ dataset card. "
            "Dừng tải để tránh tự suy đoán nhãn 0/1."
        )
    return {0: "reliable", 1: "unreliable"}


def _short_record(record: dict[str, object], max_text_length: int = 300) -> dict[str, object]:
    """Rút gọn bản ghi khi in, không truy cập/cố giải mã các cột ảnh."""
    text = str(record[ORIGINAL_TEXT_COLUMN])
    if len(text) > max_text_length:
        text = text[:max_text_length] + "..."
    return {
        ORIGINAL_TEXT_COLUMN: text,
        ORIGINAL_LABEL_COLUMN: record[ORIGINAL_LABEL_COLUMN],
    }


def download_reintel() -> Path:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print(f"Nguồn dữ liệu: https://huggingface.co/datasets/{DATASET_ID}")
    hub_info = HfApi().dataset_info(DATASET_ID)
    revision = hub_info.sha
    if not revision:
        raise RuntimeError("Hugging Face không trả về revision của bộ dữ liệu.")
    print(f"Revision đã khóa: {revision}")

    card_path = hf_hub_download(
        repo_id=DATASET_ID,
        filename="README.md",
        repo_type="dataset",
        revision=revision,
    )
    card_text = Path(card_path).read_text(encoding="utf-8")
    label_mapping = verify_label_mapping_from_card(card_text)

    print("Đang tải split train có nhãn...")
    dataset = load_dataset(DATASET_ID, split=DATASET_SPLIT, revision=revision)

    original_columns = list(dataset.column_names)
    print(f"Dataset features:\n{dataset.features}")
    print(f"Tên cột gốc ({len(original_columns)}): {original_columns}")
    print(f"Số bản ghi: {len(dataset)}")
    print("Ý nghĩa nhãn theo dataset card chính thức:")
    for value, meaning in label_mapping.items():
        print(f"- {value}: {meaning}")

    missing_required = [
        name
        for name in (ORIGINAL_TEXT_COLUMN, ORIGINAL_LABEL_COLUMN)
        if name not in original_columns
    ]
    if missing_required:
        raise RuntimeError(
            f"Thiếu cột bắt buộc {missing_required}. Lược đồ thực tế: {original_columns}"
        )

    text_and_label = dataset.select_columns([ORIGINAL_TEXT_COLUMN, ORIGINAL_LABEL_COLUMN])
    print("5 bản ghi đầu (chỉ hiện văn bản và nhãn):")
    for index in range(min(5, len(text_and_label))):
        print(json.dumps(_short_record(text_and_label[index]), ensure_ascii=False))

    dataframe = text_and_label.to_pandas()
    observed_labels = sorted(dataframe[ORIGINAL_LABEL_COLUMN].dropna().unique().tolist())
    unknown_labels = [value for value in observed_labels if value not in label_mapping]
    if unknown_labels:
        raise RuntimeError(
            f"Phát hiện nhãn chưa được dataset card giải thích: {unknown_labels}. Dừng xử lý."
        )
    print(f"Các nhãn thực tế trong split train: {observed_labels}")
    print("Phân phối nhãn thực tế:")
    print(dataframe[ORIGINAL_LABEL_COLUMN].value_counts(dropna=False).sort_index().to_string())

    # Đồ án chỉ dùng nội dung chữ và nhãn; không xuất ảnh hay metadata mạng xã hội.
    dataframe = dataframe.rename(
        columns={
            ORIGINAL_TEXT_COLUMN: PROJECT_TEXT_COLUMN,
            ORIGINAL_LABEL_COLUMN: PROJECT_LABEL_COLUMN,
        }
    )[[PROJECT_TEXT_COLUMN, PROJECT_LABEL_COLUMN]]
    dataframe.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    metadata = {
        "source": DATASET_ID,
        "source_url": f"https://huggingface.co/datasets/{DATASET_ID}",
        "source_revision": revision,
        "split": DATASET_SPLIT,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "num_records": len(dataset),
        "original_num_columns": len(original_columns),
        "original_columns": original_columns,
        "original_features": str(dataset.features),
        "original_text_column": ORIGINAL_TEXT_COLUMN,
        "original_label_column": ORIGINAL_LABEL_COLUMN,
        "project_text_column": PROJECT_TEXT_COLUMN,
        "project_label_column": PROJECT_LABEL_COLUMN,
        "label_mapping": {str(key): value for key, value in label_mapping.items()},
        "observed_labels": observed_labels,
    }
    METADATA_FILE.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Đã lưu dữ liệu văn bản có nhãn: {OUTPUT_FILE}")
    print(f"Đã lưu metadata nguồn: {METADATA_FILE}")
    return OUTPUT_FILE


if __name__ == "__main__":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
            sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
        download_reintel()
    except Exception as error:
        print(f"\nKHÔNG THỂ TẢI ReINTEL: {type(error).__name__}: {error}", file=sys.stderr)
        print(
            "Không có dữ liệu giả nào được tạo. Hãy kiểm tra Internet rồi chạy lại "
            "`python download_reintel.py`, hoặc tải split train từ trang Hugging Face "
            "và đặt CSV gồm cột text, label tại data/raw/reintel_train.csv.",
            file=sys.stderr,
        )
        raise SystemExit(1) from error
