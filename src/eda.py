"""Phân tích thăm dò cơ bản cho dữ liệu ReINTEL."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_eda_summary(
    dataframe: pd.DataFrame,
    text_column: str,
    label_column: str,
    source_metadata: dict[str, object] | None = None,
) -> str:
    missing = dataframe.isna().sum()
    label_counts = dataframe[label_column].value_counts(dropna=False)
    label_percentages = dataframe[label_column].value_counts(dropna=False, normalize=True) * 100
    valid_text = dataframe[text_column].dropna().astype(str)
    lengths = valid_text.str.len()

    lines: list[str] = [
        "TÓM TẮT PHÂN TÍCH DỮ LIỆU ReINTEL",
        "=" * 40,
    ]
    if source_metadata:
        mapping = source_metadata.get("label_mapping", {})
        lines.extend(
            [
                f"Nguồn: {source_metadata.get('source_url', source_metadata.get('source'))}",
                f"Revision nguồn: {source_metadata.get('source_revision')}",
                f"Split sử dụng: {source_metadata.get('split')}",
                f"Số cột trong dữ liệu gốc: {source_metadata.get('original_num_columns')}",
                f"Danh sách cột gốc: {source_metadata.get('original_columns')}",
                f"Cột văn bản gốc: {source_metadata.get('original_text_column')}",
                f"Cột nhãn gốc: {source_metadata.get('original_label_column')}",
                f"Ánh xạ nhãn được dataset card xác nhận: {mapping}",
                "",
            ]
        )
    lines.extend(
        [
        f"Số mẫu có nhãn trong CSV dự án: {int(dataframe[label_column].notna().sum())}",
        f"Số cột trong CSV dự án: {len(dataframe.columns)}",
        f"Danh sách cột trong CSV dự án: {list(dataframe.columns)}",
        f"Cột văn bản dùng trong dự án: {text_column}",
        f"Cột nhãn dùng trong dự án: {label_column}",
        "",
        f"Số giá trị văn bản thiếu: {int(dataframe[text_column].isna().sum())}",
        f"Số giá trị nhãn thiếu: {int(dataframe[label_column].isna().sum())}",
        "Giá trị thiếu của tất cả các cột:",
        missing.to_string(),
        "",
        f"Số dòng trùng lặp hoàn toàn: {int(dataframe.duplicated().sum())}",
        "",
        f"Các nhãn duy nhất: {dataframe[label_column].dropna().unique().tolist()}",
        f"Phân phối nhãn ({label_column}):",
        label_counts.to_string(),
        "",
        "Tỷ lệ nhãn (%):",
        label_percentages.round(2).to_string(),
        "",
        f"Thống kê độ dài văn bản theo số ký tự ({text_column}):",
        ]
    )
    if lengths.empty:
        lines.append("Không có văn bản hợp lệ để tính độ dài.")
    else:
        lines.extend(
            [
                f"Nhỏ nhất: {int(lengths.min())}",
                f"Lớn nhất: {int(lengths.max())}",
                f"Trung bình: {lengths.mean():.2f}",
                f"Trung vị: {lengths.median():.2f}",
            ]
        )
    return "\n".join(lines) + "\n"


def save_eda_report(
    dataframe: pd.DataFrame,
    text_column: str,
    label_column: str,
    output_file: Path,
    chart_file: Path | None = None,
    source_metadata: dict[str, object] | None = None,
) -> str:
    summary = build_eda_summary(dataframe, text_column, label_column, source_metadata)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(summary, encoding="utf-8")

    if chart_file is not None:
        counts = dataframe[label_column].astype("string").fillna("<thiếu>").value_counts()
        figure, axis = plt.subplots(figsize=(7, 4))
        counts.plot(kind="bar", ax=axis, color="#4472C4")
        axis.set_title("Phân phối nhãn ReINTEL")
        axis.set_xlabel("Nhãn")
        axis.set_ylabel("Số mẫu")
        axis.tick_params(axis="x", rotation=0)
        figure.tight_layout()
        chart_file.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(chart_file, dpi=150)
        plt.close(figure)
    return summary
