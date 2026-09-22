"""Chay toan bo quy trinh chuan bi ReINTEL cho cuoi tuan 3."""

from __future__ import annotations

import json
import sys

import joblib
import pandas as pd

from src import config
from src.eda import save_eda_report
from src.load_data import DataConfigurationError, load_reintel, print_dataset_preview
from src.preprocess import preprocess_dataframe, save_preprocessing_examples
from src.split_data import split_dataframe
from src.vectorize import fit_tfidf_on_train


def _prepare_valid_rows(dataframe: pd.DataFrame, text_column: str, label_column: str) -> pd.DataFrame:
    """Bo dong khong the dung va dong trung lap hoan toan truoc khi chia tap."""
    selected = dataframe.dropna(subset=[text_column, label_column]).copy()
    selected = selected[selected[text_column].astype(str).str.strip().ne("")]
    selected = selected.drop_duplicates(subset=[text_column, label_column]).copy()
    selected.insert(0, "source_index", selected.index)
    selected.reset_index(drop=True, inplace=True)
    if selected.empty:
        raise DataConfigurationError("Không còn dòng hợp lệ sau khi loại văn bản/nhãn bị thiếu.")
    return selected


def run() -> None:
    for directory in (
        config.RAW_DATA_DIR,
        config.PROCESSED_DATA_DIR,
        config.ARTIFACTS_DIR,
        config.RESULTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    print("[1/6] Đọc dữ liệu ReINTEL", flush=True)
    dataframe, data_path, text_column, label_column = load_reintel()
    metadata_path = config.RAW_DATA_DIR / "reintel_metadata.json"
    source_metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else None
    )
    print(f"Tệp dữ liệu: {data_path}")
    print(f"Cột văn bản: {text_column} | Cột nhãn: {label_column}")
    print_dataset_preview(dataframe)

    print("\n[2/6] Phân tích dữ liệu")
    summary = save_eda_report(
        dataframe,
        text_column,
        label_column,
        config.RESULTS_DIR / "eda_summary.txt",
        config.RESULTS_DIR / "label_distribution.png",
        source_metadata,
    )
    print(summary)

    print("[3/6] Tiền xử lý văn bản")
    valid_data = _prepare_valid_rows(dataframe, text_column, label_column)
    removed = len(dataframe) - len(valid_data)
    print(f"Số dòng bị loại do thiếu/rỗng/trùng lặp: {removed}")
    processed = preprocess_dataframe(valid_data, text_column)
    count_before_empty_filter = len(processed)
    processed = processed[processed["text_segmented"].str.strip().ne("")].copy()
    print(
        "Số dòng trở thành rỗng sau làm sạch và bị loại: "
        f"{count_before_empty_filter - len(processed)}"
    )
    if len(processed) < 2:
        raise DataConfigurationError("Cần ít nhất 2 văn bản hợp lệ sau tiền xử lý.")
    save_preprocessing_examples(
        processed, config.RESULTS_DIR / "preprocessing_examples.txt", count=10
    )

    print("[4/6] Chia tập huấn luyện/kiểm thử theo tỷ lệ 80/20")
    train_data, test_data = split_dataframe(processed, label_column)
    print(f"Train: {len(train_data)} mẫu | Test: {len(test_data)} mẫu")
    train_distribution = train_data[label_column].value_counts().sort_index()
    test_distribution = test_data[label_column].value_counts().sort_index()
    print("Phân phối nhãn tập train:")
    print(train_distribution.to_string())
    print("Phân phối nhãn tập test:")
    print(test_distribution.to_string())
    split_summary = "\n".join(
        [
            "TÓM TẮT CHIA DỮ LIỆU ReINTEL",
            "=" * 40,
            f"random_state: {config.RANDOM_STATE}",
            f"Tỷ lệ test: {config.TEST_SIZE:.0%}",
            f"Số mẫu train: {len(train_data)}",
            f"Số mẫu test: {len(test_data)}",
            "",
            "Phân phối nhãn train:",
            train_distribution.to_string(),
            "",
            "Phân phối nhãn test:",
            test_distribution.to_string(),
            "",
        ]
    )
    (config.RESULTS_DIR / "split_summary.txt").write_text(split_summary, encoding="utf-8")

    print("[5/6] Biểu diễn TF-IDF (chỉ fit trên tập huấn luyện)")
    vectorizer, train_tfidf, test_tfidf = fit_tfidf_on_train(
        train_data["text_segmented"], test_data["text_segmented"]
    )
    print(f"X_train_tfidf.shape: {train_tfidf.shape}")
    print(f"X_test_tfidf.shape : {test_tfidf.shape}")
    print(f"Kích thước từ vựng: {len(vectorizer.vocabulary_)}")
    print(f"Kiểu ma trận: {type(train_tfidf).__name__}")

    print("[6/6] Lưu kết quả")
    train_output = train_data[["source_index", "text_segmented", label_column]].rename(
        columns={"text_segmented": "processed_text"}
    )
    test_output = test_data[["source_index", "text_segmented", label_column]].rename(
        columns={"text_segmented": "processed_text"}
    )
    train_output.to_csv(
        config.PROCESSED_DATA_DIR / "train.csv", index=False, encoding="utf-8-sig"
    )
    test_output.to_csv(
        config.PROCESSED_DATA_DIR / "test.csv", index=False, encoding="utf-8-sig"
    )
    joblib.dump(vectorizer, config.ARTIFACTS_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(
        {
            "X_train_tfidf": train_tfidf,
            "X_test_tfidf": test_tfidf,
            "y_train": train_data[label_column].reset_index(drop=True),
            "y_test": test_data[label_column].reset_index(drop=True),
            "train_source_indices": train_data["source_index"].tolist(),
            "test_source_indices": test_data["source_index"].tolist(),
            "text_column": text_column,
            "label_column": label_column,
        },
        config.ARTIFACTS_DIR / "week3_prepared_data.joblib",
    )

    print("\nHoàn tất giai đoạn chuẩn bị dữ liệu.")
   


if __name__ == "__main__":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
            sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
        run()
    except (FileNotFoundError, DataConfigurationError, ValueError) as error:
        print(f"\nLỖI CHUẨN BỊ DỮ LIỆU: {error}", file=sys.stderr)
        raise SystemExit(1) from error
