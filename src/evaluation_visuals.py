"""Render reproducible Week 6 figures and an offline Vietnamese test report.

This module only presents a completed evaluation summary. It never loads a
model, modifies a dataset, or chooses a model using test-set measurements.
"""

from __future__ import annotations

from html import escape
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import PercentFormatter
import numpy as np


_BLUE = "#2563eb"
_TEAL = "#0d9488"
_AMBER = "#d97706"
_RED = "#dc2626"
_INK = "#172b4d"
_MUTED = "#52647c"
_LABELS = ["0 · Đáng tin cậy", "1 · Không đáng tin cậy"]
_METRICS = [
    ("accuracy", "Accuracy"),
    ("precision", "Precision · nhãn 1"),
    ("recall", "Recall · nhãn 1"),
    ("f1", "F1 · nhãn 1"),
]


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", value)


def _pct(value: float) -> str:
    return f"{100 * float(value):.2f}%"


def _number(value: int) -> str:
    return f"{int(value):,}".replace(",", ".")


def _short_name(model: dict) -> str:
    model_id = model["id"].lower()
    kernel = "Linear" if "linear" in model_id else "RBF" if "rbf" in model_id else model["display_name"]
    stage = "cơ sở" if "baseline" in model_id else "Grid Search"
    return f"{kernel}\n{stage}"


def _model_title(model: dict, selected: str) -> str:
    suffix = " · chọn theo CV" if model["id"] == selected else ""
    return f"{model['display_name']}{suffix}"


def _save(fig: plt.Figure, figure_dir: Path, name: str) -> None:
    fig.savefig(figure_dir / f"{name}.png", dpi=170, facecolor="white", bbox_inches="tight")
    fig.savefig(figure_dir / f"{name}.svg", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _matrix_axis(ax: plt.Axes, model: dict, normalized: bool, vmax: float | None = None) -> None:
    matrix = np.asarray(model["confusion_matrix"], dtype=float)
    if normalized:
        totals = matrix.sum(axis=1, keepdims=True)
        values = np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals != 0)
        vmax = 1.0
    else:
        values = matrix
        vmax = vmax or float(matrix.max()) or 1.0
    ax.imshow(values, cmap="Blues", vmin=0, vmax=vmax)
    codes = np.array([["TN", "FP"], ["FN", "TP"]])
    for row in range(2):
        for column in range(2):
            value = values[row, column]
            main = f"{value:.1%}" if normalized else _number(int(value))
            count = f"\n{_number(int(matrix[row, column]))} mẫu" if normalized else ""
            ax.text(
                column, row, f"{codes[row, column]}\n{main}{count}",
                ha="center", va="center", fontsize=12,
                color="white" if value > 0.48 * vmax else _INK,
                fontweight="bold",
            )
    ax.set_xticks([0, 1], ["0 · Đáng tin cậy", "1 · Không đáng\ntin cậy"], fontsize=10)
    ax.set_yticks([0, 1], ["0 · Đáng\ntin cậy", "1 · Không đáng\ntin cậy"], fontsize=10)
    ax.set_xlabel("Nhãn dự đoán", labelpad=8)
    ax.set_ylabel("Nhãn thực", labelpad=8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_confusion_matrices(summary: dict, figure_dir: Path) -> None:
    selected = summary["selected_model_id"]
    for model in summary["models"]:
        for normalized in (False, True):
            fig, ax = plt.subplots(figsize=(6.2, 5.3))
            _matrix_axis(ax, model, normalized)
            subtitle = "Tỷ lệ theo từng nhãn thực" if normalized else "Số lượng mẫu"
            ax.set_title(f"{_model_title(model, selected)}\n{subtitle}", fontsize=13, pad=18)
            fig.tight_layout(rect=(0, 0.07, 1, 1))
            fig.text(0.5, 0.015, "Nhãn dương: 1 = Không đáng tin cậy", ha="center", fontsize=9, color=_MUTED)
            suffix = "normalized" if normalized else "counts"
            _save(fig, figure_dir, f"confusion_{_slug(model['id'])}_{suffix}")

    models = summary["models"]
    rows = (len(models) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(11.8, rows * 4.5), squeeze=False)
    max_count = max(int(np.max(model["confusion_matrix"])) for model in models)
    for ax, model in zip(axes.flat, models):
        _matrix_axis(ax, model, False, max_count)
        ax.set_title(_model_title(model, selected), fontsize=12, pad=12)
    for ax in list(axes.flat)[len(models):]:
        ax.axis("off")
    fig.suptitle("Ma trận nhầm lẫn trên cùng tập kiểm thử", fontsize=18, fontweight="bold", y=1.015)
    fig.text(0.5, -0.01, "Hàng = nhãn thực · Cột = nhãn dự đoán · Các ma trận dùng cùng thang màu", ha="center", color=_MUTED)
    fig.tight_layout(h_pad=3, w_pad=2)
    _save(fig, figure_dir, "confusion_overview")


def _clean_axis(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.18)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")


def _draw_metrics(summary: dict, figure_dir: Path) -> None:
    models = summary["models"]
    x = np.arange(len(models))
    colors = [_TEAL if model["id"] == summary["selected_model_id"] else _BLUE for model in models]
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.4))
    for ax, (key, label) in zip(axes.flat, _METRICS):
        values = [model["metrics"][key] for model in models]
        bars = ax.bar(x, values, color=colors, width=0.64, zorder=3)
        ax.bar_label(bars, labels=[_pct(v) for v in values], padding=5, fontsize=10, fontweight="bold")
        ax.set_xticks(x, [_short_name(model) for model in models], fontsize=10)
        ax.set_ylim(0, 1.13)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_title(label, fontsize=13, fontweight="bold", pad=12)
        _clean_axis(ax)
    fig.suptitle("Hiệu năng của bốn cấu hình SVM trên tập test", fontsize=18, fontweight="bold", y=1.02)
    fig.text(0.5, -0.02, "Màu xanh ngọc: mô hình đã chọn theo CV ở Tuần 4–5 · Precision, Recall và F1 tính cho nhãn 1", ha="center", color=_MUTED, fontsize=10)
    fig.tight_layout(h_pad=3, w_pad=2)
    _save(fig, figure_dir, "metric_comparison")


def _draw_errors(summary: dict, figure_dir: Path) -> None:
    models = summary["models"]
    fig, ax = plt.subplots(figsize=(12.4, 5.4))
    y = np.arange(len(models))
    left = np.zeros(len(models))
    categories = [
        (0, 0, "TN · Đúng nhãn 0", "#94a3b8"),
        (1, 1, "TP · Đúng nhãn 1", _TEAL),
        (0, 1, "FP · Báo nhầm nhãn 1", _AMBER),
        (1, 0, "FN · Bỏ sót nhãn 1", _RED),
    ]
    for row, column, label, color in categories:
        values = np.array([model["confusion_matrix"][row][column] for model in models])
        bars = ax.barh(y, values, left=left, color=color, height=0.61, label=label)
        for bar, value in zip(bars, values):
            if value >= 18:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                        str(value), va="center", ha="center", color="white", fontsize=11, fontweight="bold")
        left += values
    ax.set_yticks(y, [model["display_name"] for model in models])
    ax.invert_yaxis()
    ax.set_xlim(0, max(left) * 1.015)
    ax.set_xlabel("Số mẫu kiểm thử")
    ax.set_title("Phân bố dự đoán đúng và hai loại lỗi", fontsize=17, fontweight="bold", pad=18)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
    _clean_axis(ax)
    ax.grid(False, axis="y")
    ax.grid(axis="x", alpha=0.18)
    fig.tight_layout()
    _save(fig, figure_dir, "error_breakdown")


def _draw_distribution(summary: dict, figure_dir: Path) -> None:
    labels = summary["data_audit"]["evaluated_test_labels"]
    models = summary["models"]
    names = ["Nhãn thực"] + [_short_name(model) for model in models]
    counts = [[int(labels["0"]), int(labels["1"])]]
    counts.extend(np.asarray(model["confusion_matrix"]).sum(axis=0).tolist() for model in models)
    values = np.asarray(counts)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(12.4, 5.3))
    for index, color in enumerate((_BLUE, _TEAL)):
        bars = ax.bar(x + (index - 0.5) * 0.35, values[:, index], 0.35, label=_LABELS[index], color=color)
        ax.bar_label(bars, padding=4, fontsize=10)
    ax.set_xticks(x, names)
    ax.set_ylim(0, max(values.flatten()) * 1.16)
    ax.set_ylabel("Số mẫu")
    ax.set_title("Nhãn thực và phân bố nhãn dự đoán", fontsize=17, fontweight="bold", pad=18)
    ax.legend(frameon=False, loc="upper right")
    _clean_axis(ax)
    fig.tight_layout()
    _save(fig, figure_dir, "predicted_label_distribution")


def _draw_cv_test(summary: dict, figure_dir: Path) -> None:
    models = summary["models"]
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(12.4, 5.3))
    for offset, key, label, color in [(-0.18, "cv_f1", "CV trên train · Tuần 4–5", "#64748b"),
                                     (0.18, "f1", "Test độc lập · Tuần 6", _TEAL)]:
        values = [model["cv_f1"] if key == "cv_f1" else model["metrics"][key] for model in models]
        bars = ax.bar(x + offset, values, 0.35, label=label, color=color)
        ax.bar_label(bars, labels=[_pct(value) for value in values], padding=4, fontsize=10)
    ax.set_xticks(x, [_short_name(model) for model in models])
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_ylabel("F1 của nhãn 1")
    ax.set_title("Đối chiếu F1 giữa Cross Validation và tập test", fontsize=17, fontweight="bold", pad=18)
    ax.legend(frameon=False, loc="upper right")
    _clean_axis(ax)
    fig.text(0.5, -0.04, "CV là trung bình các folds trên train; test là một tập dữ liệu riêng. Điểm test không dùng để chọn lại mô hình.",
             ha="center", fontsize=10, color=_MUTED)
    fig.tight_layout()
    _save(fig, figure_dir, "cv_vs_test_f1")


def _draw_flow(summary: dict, figure_dir: Path) -> None:
    audit = summary["data_audit"]
    boxes = [
        ("01 · Cố định mô hình", "4 cấu hình từ Tuần 4–5\nChọn mô hình theo F1 CV"),
        ("02 · Kiểm tra test", f"{_number(audit['original_test_rows'])} mẫu ban đầu\nLoại {_number(audit['excluded_rows'])} mẫu trùng"),
        ("03 · Dự đoán", f"{_number(audit['evaluated_test_rows'])} mẫu đánh giá\nTF-IDF → SVM đã lưu"),
        ("04 · Thống kê", "Accuracy · Precision\nRecall · F1 · Ma trận"),
        ("05 · Xuất kết quả", "CSV · PNG · SVG\nBáo cáo HTML ngoại tuyến"),
    ]
    fig, ax = plt.subplots(figsize=(15.2, 3.1))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 3)
    ax.axis("off")
    for index, (title, body) in enumerate(boxes):
        x = index * 3 + 0.05
        patch = FancyBboxPatch((x, 0.7), 2.7, 1.7, boxstyle="round,pad=0.04,rounding_size=0.12",
                               facecolor="#eff6ff" if index != 4 else "#ecfdf5", edgecolor="#bfdbfe")
        ax.add_patch(patch)
        ax.text(x + 1.35, 2.04, title, ha="center", fontsize=12, fontweight="bold", color=_INK)
        ax.text(x + 1.35, 1.38, body, ha="center", va="center", fontsize=11, color=_MUTED, linespacing=1.6)
        if index < len(boxes) - 1:
            ax.add_patch(FancyArrowPatch((x + 2.73, 1.55), (x + 2.98, 1.55), arrowstyle="-|>", mutation_scale=14, color=_BLUE))
    ax.text(7.5, 2.8, "Quy trình đánh giá Tuần 6", ha="center", fontsize=18, fontweight="bold", color=_INK)
    ax.text(7.5, 0.2, "Không huấn luyện lại · Không dùng kết quả test để chọn tham số", ha="center", fontsize=11, color=_MUTED)
    _save(fig, figure_dir, "evaluation_flow")


def _html_table(headers: list[str], rows: list[list[str]], numeric_from: int = 1) -> str:
    header = "".join(f"<th scope='col'>{escape(str(value))}</th>" for value in headers)
    body_rows = []
    for row in rows:
        cells = []
        for index, value in enumerate(row):
            attributes = ' class="num"' if index >= numeric_from else ""
            cells.append(f"<td{attributes}>{value}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    body = "".join(body_rows)
    return f"<div class='table-scroll'><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>"


def _chart_html(name: str, title: str, description: str, wide: bool = False) -> str:
    return f"""<figure class="chart{' wide' if wide else ''}">
      <figcaption><h3>{escape(title)}</h3><p>{escape(description)}</p></figcaption>
      <a href="figures/{name}.png" target="_blank"><img src="figures/{name}.png" alt="{escape(title)}" loading="lazy"></a>
      <div class="figure-links"><a href="figures/{name}.png" download>PNG</a><a href="figures/{name}.svg" download>SVG</a></div>
    </figure>"""


def _class_report_rows(report: dict) -> list[list[str]]:
    rows = []
    for names, label in [(("reliable", "0"), "0 · Đáng tin cậy"),
                         (("unreliable", "1"), "1 · Không đáng tin cậy"),
                         (("macro avg",), "Trung bình macro"),
                         (("weighted avg",), "Trung bình có trọng số")]:
        entry = next((report[name] for name in names if name in report), None)
        if entry is None:
            continue
        rows.append([escape(label), _pct(entry["precision"]), _pct(entry["recall"]),
                     _pct(entry["f1-score"]), _number(entry["support"])])
    return rows


def _params_text(params: dict) -> str:
    return "; ".join(f"{key.removeprefix('svc__')}={value}" for key, value in params.items())


def _model_href(model: dict) -> str:
    raw = str(model["model_file"]).replace("\\", "/")
    if "artifacts/" in raw:
        return "../artifacts/" + raw.split("artifacts/", 1)[1]
    return "../" + raw.lstrip("/")


def _write_html(summary: dict, docs_dir: Path) -> None:
    audit = summary["data_audit"]
    models = summary["models"]
    selected = next(model for model in models if model["id"] == summary["selected_model_id"])
    selected_metrics = selected["metrics"]
    rows = []
    for model in models:
        name = escape(model["display_name"])
        if model["id"] == selected["id"]:
            name += " <span class='badge'>Chọn theo CV</span>"
        rows.append([name, *[_pct(model["metrics"][key]) for key, _ in _METRICS],
                     _pct(model["metrics"]["f1_macro"]), _pct(model["metrics"]["f1_weighted"]),
                     _pct(model["metrics"]["balanced_accuracy"])])
    metric_table = _html_table(["Cấu hình SVM", "Accuracy", "Precision¹", "Recall¹", "F1¹", "F1 macro", "F1 weighted", "Balanced accuracy"], rows)
    audit_rows = [
        ["Tập train đã dùng ở Tuần 4–5", _number(audit["train_rows"])],
        ["Tập test ban đầu", _number(audit["original_test_rows"])],
        ["Loại vì trùng văn bản với train", _number(audit["excluded_overlap_rows"])],
        ["Loại vì trùng trong test còn lại", _number(audit["excluded_duplicate_rows"])],
        ["Tổng mẫu bị loại", _number(audit["excluded_rows"])],
        ["Tập test được đánh giá", _number(audit["evaluated_test_rows"])],
        ["Nhãn 0 · Đáng tin cậy", _number(audit["evaluated_test_labels"]["0"])],
        ["Nhãn 1 · Không đáng tin cậy", _number(audit["evaluated_test_labels"]["1"])],
    ]
    audit_table = _html_table(["Thống kê dữ liệu", "Số mẫu"], audit_rows)
    matrix_rows = []
    cv_rows = []
    configuration_rows = []
    model_cards = []
    for model in models:
        tn, fp = model["confusion_matrix"][0]
        fn, tp = model["confusion_matrix"][1]
        matrix_rows.append([escape(model["display_name"]), _number(tn), _number(fp), _number(fn), _number(tp), _number(fp + fn)])
        cv_rows.append([escape(model["display_name"]), _pct(model["cv_f1"]), _pct(model["metrics"]["f1"]),
                        f"{100 * (model['metrics']['f1'] - model['cv_f1']):+.2f} điểm %"])
        configuration_rows.append([escape(model["display_name"]), escape(_params_text(model["params"]))])
        slug = _slug(model["id"])
        report_table = _html_table(["Nhãn / cách lấy trung bình", "Precision", "Recall", "F1", "Support"], _class_report_rows(model["classification_report"]))
        selected_badge = "<span class='badge'>Mô hình chọn theo CV</span>" if model["id"] == selected["id"] else ""
        model_cards.append(f"""<article class="model-card" id="model-{slug}" data-model="{escape(model['id'])}">
          <div class="model-heading"><h3>{escape(model['display_name'])}</h3>{selected_badge}</div>
          <p class="params">{escape(_params_text(model['params']))}</p>
          <div class="matrix-pair">
            {_chart_html(f'confusion_{slug}_counts', 'Ma trận nhầm lẫn · số mẫu', 'TN / FP ở hàng nhãn 0; FN / TP ở hàng nhãn 1.')}
            {_chart_html(f'confusion_{slug}_normalized', 'Ma trận nhầm lẫn · chuẩn hóa', 'Mỗi hàng cộng thành 100%; chuẩn hóa theo nhãn thực.')}
          </div>
          <h4>Báo cáo phân loại từng nhãn</h4>{report_table}
          <div class="downloads"><a href="{escape(model['prediction_file'])}">Dự đoán từng mẫu · CSV</a><a href="{escape(model['error_file'])}">Các mẫu dự đoán sai · CSV</a><a href="{escape(_model_href(model))}">Mô hình .pkl</a></div>
          <details><summary>Định danh mô hình và SHA-256</summary><p><code>{escape(str(model['model_file']))}</code></p><code class="hash">{escape(model['model_sha256'])}</code></details>
        </article>""")
    matrix_table = _html_table(["Cấu hình SVM", "TN", "FP", "FN", "TP", "Tổng lỗi"], matrix_rows)
    cv_table = _html_table(["Cấu hình SVM", "F1 CV", "F1 test", "Test − CV"], cv_rows)
    configuration_table = _html_table(["Cấu hình SVM", "Tham số đã cố định"], configuration_rows, numeric_from=99)
    chart_blocks = "".join([
        _chart_html("metric_comparison", "Accuracy, Precision, Recall và F1", "Bốn cấu hình sử dụng cùng tập test; màu xanh ngọc đánh dấu cấu hình chọn theo CV.", True),
        _chart_html("error_breakdown", "Dự đoán đúng và các loại lỗi", "FP: nhãn 0 bị báo thành 1. FN: nhãn 1 bị bỏ sót thành 0."),
        _chart_html("predicted_label_distribution", "Phân bố nhãn dự đoán", "Đối chiếu số nhãn dự đoán của từng cấu hình với nhãn thực."),
        _chart_html("cv_vs_test_f1", "F1 trên CV và test", "CV là trung bình trên train; test được đánh giá sau khi đã cố định mô hình.", True),
    ])
    cards = "".join(f"<div class='stat'><span>{label}</span><strong>{_pct(selected_metrics[key])}</strong></div>" for key, label in _METRICS)
    table_titles = {"metrics": "Bảng chỉ số hiệu năng", "classification_report": "Báo cáo phân loại chi tiết",
                    "confusion_matrices": "Bảng TN / FP / FN / TP", "data_audit": "Danh sách dòng test bị loại",
                    "evaluated_test_samples": "Tập test sau kiểm tra",
                    "train_label_conflicts": "Văn bản train có nhãn mâu thuẫn",
                    "predictions": "Toàn bộ dự đoán của bốn cấu hình"}
    table_links = "".join(f"<a href='{escape(path)}'>{escape(table_titles.get(key, key))} · CSV</a>" for key, path in summary["tables"].items())
    versions = " · ".join(f"{escape(str(name))} {escape(str(version))}" for name, version in summary["versions"].items())
    options = "".join(f"<option value='{escape(model['id'])}'>{escape(model['display_name'])}</option>" for model in models)
    conflict_note = ""
    if audit.get("train_text_conflict_groups", 0):
        conflict_note = (
            f"<p class='note'>Train có {audit['train_text_conflict_groups']} nhóm văn bản sau tiền xử lý "
            "giống nhau nhưng gán cả nhãn 0 và 1. Đây là giới hạn chất lượng nhãn cần nêu trong đồ án; "
            "lần đánh giá này sử dụng nguyên mô hình đã huấn luyện. "
            f"<a href='{escape(summary['tables']['train_label_conflicts'])}'>Xem các dòng nhãn mâu thuẫn · CSV</a>.</p>"
        )
    content = f"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Đồ án SVM · Kết quả đánh giá Tuần 6</title>
<style>
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:#f3f6fa;color:#172b4d;font:15px/1.65 'Segoe UI',Arial,sans-serif}}a{{color:#1d4ed8;text-underline-offset:3px}}h1,h2,h3,h4{{line-height:1.3}}h1{{font-size:clamp(27px,4vw,40px);margin:9px 0 16px;letter-spacing:-.7px}}h2{{font-size:25px;margin:0 0 16px}}h3{{font-size:18px;margin:0 0 8px}}h4{{margin:22px 0 12px}}p{{margin:10px 0;color:#52647c}}.page{{max-width:1280px;margin:auto;padding:38px 32px 60px}}.eyebrow{{font-size:12px;font-weight:700;letter-spacing:1.8px;text-transform:uppercase;color:#2563eb}}.intro{{max-width:820px}}.meta{{font-size:13px;color:#64748b}}nav{{display:flex;gap:10px 23px;flex-wrap:wrap;margin:25px 0 28px;padding:15px 0;border-top:1px solid #dbe3ef;border-bottom:1px solid #dbe3ef}}nav a{{text-decoration:none;font-size:14px;font-weight:600}}section{{scroll-margin-top:24px;margin-bottom:32px;background:white;border:1px solid #e0e7ef;border-radius:16px;padding:28px}}.selection{{background:#eff9f6;border:1px solid #b6e3d5;border-radius:13px;padding:19px 23px;margin-bottom:18px}}.selection p{{margin:4px 0;color:#315749}}.badge{{display:inline-block;border-radius:20px;padding:3px 9px;background:#dff6eb;color:#116149;font-size:11px;font-weight:700;white-space:nowrap}}.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:20px 0}}.stat{{background:#f7f9fc;border:1px solid #e5ebf3;border-radius:12px;padding:16px 20px}}.stat span{{display:block;font-size:13px;color:#52647c}}.stat strong{{font-size:29px;color:#15355c;font-variant-numeric:tabular-nums}}.note{{font-size:13px;background:#f7f9fd;padding:12px 15px;border-left:3px solid #94a3b8}}.table-scroll{{overflow-x:auto;margin:15px 0}}table{{border-collapse:collapse;width:100%;font-size:13px}}th{{background:#edf3fb;color:#344967;text-align:left;white-space:nowrap;font-weight:600}}th,td{{padding:12px 13px;border-bottom:1px solid #e6ebf2}}tbody tr:nth-child(even){{background:#fafbfd}}tbody tr:hover{{background:#f0f6ff}}.num{{text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}}.charts,.matrix-pair{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}.chart{{margin:0;background:#fff;border:1px solid #e5ebf3;border-radius:12px;padding:18px;overflow:hidden}}.chart p{{font-size:13px;margin-bottom:14px}}.chart img{{display:block;max-width:100%;height:auto;width:100%;margin:auto}}.wide{{grid-column:1/-1}}.figure-links{{display:flex;justify-content:flex-end;gap:16px;margin-top:7px;font-size:12px}}.model-card{{border:1px solid #dfe7f1;border-radius:13px;padding:22px;margin-top:23px}}.model-heading{{display:flex;gap:12px;flex-wrap:wrap;align-items:center}}.model-heading h3{{margin:0}}.params{{font-size:13px;margin-bottom:18px;overflow-wrap:anywhere}}.downloads{{display:flex;gap:10px 23px;flex-wrap:wrap;font-size:13px;margin:18px 0}}.file-grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}}.file-grid a{{display:block;padding:14px 18px;border:1px solid #dfe7f1;border-radius:9px;background:#f9fbfe}}details{{font-size:12px;color:#52647c;border-top:1px solid #e6ebf2;padding-top:12px}}summary{{cursor:pointer}}code{{font-family:Consolas,monospace;overflow-wrap:anywhere}}.hash{{display:block;word-break:break-all}}.model-filter{{display:flex;gap:12px;align-items:center;font-size:13px;margin:20px 0}}select{{font:inherit;padding:8px 12px;border:1px solid #cbd5e1;border-radius:6px;background:white;color:#172b4d}}.two-column{{display:grid;grid-template-columns:1fr 1.2fr;gap:25px}}.formulas{{font-size:14px;color:#52647c}}footer{{font-size:12px;color:#64748b;padding:0 5px}}.hidden{{display:none}}
@media(max-width:850px){{.page{{padding:22px 16px}}section{{padding:20px 16px}}.stats{{grid-template-columns:repeat(2,1fr)}}.charts,.matrix-pair,.two-column,.file-grid{{grid-template-columns:1fr}}.wide{{grid-column:auto}}.model-card{{padding:15px}}.stat strong{{font-size:25px}}.model-filter{{align-items:flex-start;flex-direction:column}}}}
@media print{{body{{background:white}}.page{{padding:0;max-width:none}}section{{break-inside:avoid;border-radius:0}}nav,.figure-links,.model-filter{{display:none}}.model-card{{break-before:page}}.hidden{{display:block}}a{{color:inherit;text-decoration:none}}}}
</style></head><body><main class="page">
<header><div class="eyebrow">Đồ án chuyên ngành · Support Vector Machine</div>
<h1>Kết quả đánh giá Tuần 6</h1>
<p class="intro">Vận hành bốn cấu hình SVM đã lưu trên tập kiểm thử; tổng hợp chỉ số hiệu năng, ma trận nhầm lẫn và dự đoán từng mẫu. Tất cả bảng và biểu đồ được sinh từ cùng một lần đánh giá.</p>
<p class="meta">Hoàn tất: {escape(summary['completed_at'])} · {_number(audit['evaluated_test_rows'])} mẫu test · {len(models)} cấu hình SVM</p></header>
<nav aria-label="Các mục báo cáo"><a href="#overview">Tổng quan</a><a href="#data">Dữ liệu</a><a href="#charts">Biểu đồ</a><a href="#models">Từng cấu hình</a><a href="#exports">Tệp kết quả</a><a href="#method">Cách tính</a></nav>
<section id="overview"><h2>Chỉ số trên tập kiểm thử</h2>
<div class="selection"><strong>{escape(selected['display_name'])} <span class="badge">Đã chọn theo CV ở Tuần 4–5</span></strong><p>{escape(_params_text(selected['params']))}</p><p>F1 CV: {_pct(selected['cv_f1'])}. Kết quả test dưới đây dùng để đánh giá; mô hình đã chọn được giữ nguyên.</p></div>
<div class="stats">{cards}</div>{metric_table}
<p class="note">¹ Precision, Recall và F1 tính cho nhãn dương <strong>1 = Không đáng tin cậy</strong>. Nhãn 0 = Đáng tin cậy. Accuracy tính trên toàn bộ tập test. <code>svm_best_model.pkl</code> là bản lưu của cấu hình chọn theo CV; báo cáo chỉ đếm bốn cấu hình SVM riêng biệt.</p>
<h3>Tham số của các cấu hình đã lưu</h3>{configuration_table}</section>
<section id="data"><h2>Tập kiểm thử và quy trình đánh giá</h2><div class="two-column"><div>{audit_table}</div><div><h3>Tính độc lập của tập test</h3><p>Đối chiếu văn bản sau tiền xử lý với train, loại các dòng trùng train và các dòng lặp trong phần test còn lại. Dữ liệu gốc được giữ nguyên; danh sách dòng bị loại và tập test được giữ được lưu riêng.</p><p>Các mô hình đã huấn luyện chỉ thực hiện transform TF-IDF và predict. Không fit lại vectorizer hoặc SVM trên test; mọi cấu hình dùng đúng cùng tập test sau kiểm tra.</p><p><a href="{escape(summary['tables']['data_audit'])}">Các dòng test bị loại · CSV</a> · <a href="{escape(summary['tables']['evaluated_test_samples'])}">Tập test được đánh giá · CSV</a></p></div></div>{conflict_note}
{_chart_html('evaluation_flow', 'Quy trình từ mô hình đã lưu đến kết quả', 'Cố định mô hình trước khi đọc chỉ số test; xuất đầy đủ bảng và hình để sử dụng trong báo cáo.', True)}</section>
<section id="charts"><h2>So sánh các cấu hình</h2><div class="charts">{chart_blocks}</div><h3 style="margin-top:25px">Bảng số lượng dự đoán đúng và sai</h3>{matrix_table}<h3 style="margin-top:25px">Đối chiếu F1 CV và test</h3>{cv_table}<p class="note">Chênh lệch tính bằng điểm phần trăm. CV và test có cách lấy mẫu khác nhau; bảng này mô tả kết quả, không dùng để xếp hạng lại hoặc chọn tham số.</p>
{_chart_html('confusion_overview', 'Tổng quan bốn ma trận nhầm lẫn', 'Cùng thang màu số lượng mẫu. Hàng là nhãn thực; cột là nhãn dự đoán.', True)}</section>
<section id="models"><h2>Ma trận và báo cáo theo từng cấu hình</h2><p>Mỗi cấu hình có ma trận số lượng, ma trận chuẩn hóa theo hàng, bảng chỉ số từng nhãn và tệp dự đoán từng mẫu.</p>
<div class="model-filter"><label for="model-select">Hiển thị cấu hình:</label><select id="model-select"><option value="all">Tất cả bốn cấu hình</option>{options}</select></div>{''.join(model_cards)}</section>
<section id="exports"><h2>Tệp số liệu để đưa vào đồ án</h2><div class="file-grid">{table_links}<a href="evaluation_summary.json">Kết quả đầy đủ · JSON</a><a href="README.md">Báo cáo Tuần 6 · Markdown</a><a href="figures/">Thư mục biểu đồ · PNG và SVG</a></div><p>CSV được ghi UTF-8 có BOM để mở thuận tiện trong Excel. PNG dùng cho báo cáo và slide; SVG giữ độ nét khi phóng to. Báo cáo HTML này chạy ngoại tuyến, không cần máy chủ hay thư viện từ Internet.</p></section>
<section id="method"><h2>Quy ước và cách tính</h2><div class="two-column"><div class="formulas"><p><strong>TN:</strong> Nhãn thực 0, dự đoán 0. <strong>FP:</strong> Nhãn thực 0, dự đoán 1.</p><p><strong>FN:</strong> Nhãn thực 1, dự đoán 0. <strong>TP:</strong> Nhãn thực 1, dự đoán 1.</p><p>Accuracy = (TP + TN) / (TN + FP + FN + TP)</p><p>Precision = TP / (TP + FP)<br>Recall = TP / (TP + FN)<br>F1 = 2 × Precision × Recall / (Precision + Recall)</p></div><div><p>Ma trận chuẩn hóa chia từng ô cho tổng hàng tương ứng. Vì vậy ô TP của hàng nhãn 1 thể hiện Recall, và ô FN thể hiện tỷ lệ bỏ sót nhãn 1.</p><p>F1 macro là trung bình F1 hai nhãn; F1 weighted có trọng số theo số mẫu từng nhãn. Balanced accuracy là trung bình Recall hai nhãn. Khi mẫu số của Precision, Recall hoặc F1 bằng 0, chỉ số được quy ước là 0.</p><p>Các nhãn ReINTEL mô tả mức độ đáng tin cậy của bài đăng; báo cáo đánh giá khả năng phân loại theo nhãn của bộ dữ liệu.</p></div></div></section>
<footer><p>Chạy lại từ thư mục dự án: <code>.\\.venv\\Scripts\\python.exe run_evaluation.py</code></p><p>{versions}</p><p>Tệp mô hình và dữ liệu được định danh trong <a href="evaluation_summary.json">evaluation_summary.json</a>.</p></footer>
</main><script>
document.getElementById('model-select').addEventListener('change',function(){{document.querySelectorAll('.model-card').forEach(function(card){{card.classList.toggle('hidden',this.value!=='all'&&card.dataset.model!==this.value);}},this);}});
</script></body></html>"""
    (docs_dir / "index.html").write_text(content, encoding="utf-8")


def _write_readme(summary: dict, docs_dir: Path) -> None:
    audit = summary["data_audit"]
    selected = next(model for model in summary["models"] if model["id"] == summary["selected_model_id"])
    metric_rows = []
    matrix_rows = []
    cv_rows = []
    details = []
    for model in summary["models"]:
        title = model["display_name"] + (" **(chọn theo CV)**" if model["id"] == selected["id"] else "")
        metric_rows.append(f"| {title} | " + " | ".join(_pct(model["metrics"][key]) for key, _ in _METRICS) +
                           f" | {_pct(model['metrics']['f1_macro'])} | {_pct(model['metrics']['f1_weighted'])} | {_pct(model['metrics']['balanced_accuracy'])} |")
        tn, fp = model["confusion_matrix"][0]
        fn, tp = model["confusion_matrix"][1]
        matrix_rows.append(f"| {model['display_name']} | {tn} | {fp} | {fn} | {tp} | {fp + fn} |")
        cv_rows.append(f"| {model['display_name']} | {_pct(model['cv_f1'])} | {_pct(model['metrics']['f1'])} | {100 * (model['metrics']['f1'] - model['cv_f1']):+.2f} |")
        class_rows = ["| " + " | ".join(row) + " |" for row in _class_report_rows(model["classification_report"])]
        slug = _slug(model["id"])
        details.append(f"""### {model['display_name']}

Tham số: `{_params_text(model['params'])}`.

| Nhãn / cách lấy trung bình | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
{chr(10).join(class_rows)}

![Ma trận số lượng – {model['display_name']}](figures/confusion_{slug}_counts.png)

![Ma trận chuẩn hóa theo hàng – {model['display_name']}](figures/confusion_{slug}_normalized.png)

[Dự đoán từng mẫu]({model['prediction_file']}) · [Các mẫu dự đoán sai]({model['error_file']}) · [Mô hình .pkl]({_model_href(model)})
""")
    table_titles = {"metrics": "Bảng chỉ số hiệu năng", "classification_report": "Báo cáo phân loại chi tiết",
                    "confusion_matrices": "Bảng TN / FP / FN / TP", "data_audit": "Danh sách dòng test bị loại",
                    "evaluated_test_samples": "Tập test sau kiểm tra", "train_label_conflicts": "Văn bản train có nhãn mâu thuẫn",
                    "predictions": "Toàn bộ dự đoán của bốn cấu hình"}
    csv_links = "\n".join(f"- [{table_titles.get(key, key)}]({path})" for key, path in summary["tables"].items())
    conflict_note = ""
    if audit.get("train_text_conflict_groups", 0):
        conflict_note = (
            f"Train có **{audit['train_text_conflict_groups']} nhóm văn bản sau tiền xử lý giống nhau nhưng nhãn mâu thuẫn (0/1)**. "
            "Đây là giới hạn chất lượng nhãn cần nêu trong báo cáo; lần đánh giá này giữ nguyên các mô hình đã huấn luyện. "
            f"Xem [các dòng có nhãn mâu thuẫn]({summary['tables']['train_label_conflicts']})."
        )
    content = f"""# Báo cáo đánh giá mô hình SVM – Tuần 6

**Đã hoàn thành:** dự đoán trên tập test, xuất dự đoán từng mẫu, tính Accuracy / Precision / Recall / F1, trích xuất và vẽ ma trận nhầm lẫn, so sánh trực quan bốn cấu hình SVM đã lưu.

Lần đánh giá hoàn tất: **{summary['completed_at']}**. Mở [báo cáo HTML](index.html) bằng trình duyệt để xem toàn bộ bảng và hình, hoặc xem các bảng dưới đây.

## 1. Tập kiểm thử

| Nội dung | Số mẫu |
|---|---:|
| Train đã dùng ở Tuần 4–5 | {audit['train_rows']} |
| Test ban đầu | {audit['original_test_rows']} |
| Loại vì trùng văn bản sau tiền xử lý với train | {audit['excluded_overlap_rows']} |
| Loại vì trùng trong phần test còn lại | {audit['excluded_duplicate_rows']} |
| Tổng số dòng bị loại | {audit['excluded_rows']} |
| **Test được đánh giá** | **{audit['evaluated_test_rows']}** |
| Nhãn 0 – Đáng tin cậy | {audit['evaluated_test_labels']['0']} |
| Nhãn 1 – Không đáng tin cậy | {audit['evaluated_test_labels']['1']} |

Dữ liệu gốc được giữ nguyên. [Danh sách dòng bị loại]({summary['tables']['data_audit']}) và [tập test được giữ]({summary['tables']['evaluated_test_samples']}) được lưu riêng. Mọi cấu hình được đánh giá trên cùng tập test sau kiểm tra. Pipeline đã lưu chỉ transform TF-IDF và predict; không fit lại trên test.

{conflict_note}

![Quy trình đánh giá](figures/evaluation_flow.png)

## 2. Mô hình đã chọn và chỉ số hiệu năng

Mô hình **{selected['display_name']}** được chọn theo **F1 CV = {_pct(selected['cv_f1'])}** từ Tuần 4–5, trước khi đánh giá test. Tham số: `{_params_text(selected['params'])}`. Kết quả test dùng để đánh giá và không dùng để chọn lại mô hình hoặc tham số.

Nhãn dương là **1 = Không đáng tin cậy**; Precision, Recall và F1 dưới đây tính cho nhãn 1. Accuracy tính trên tất cả mẫu.

| Cấu hình SVM | Accuracy | Precision nhãn 1 | Recall nhãn 1 | F1 nhãn 1 | F1 macro | F1 weighted | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(metric_rows)}

`svm_best_model.pkl` là bản lưu của cấu hình chọn theo CV, không phải cấu hình thứ năm.

![So sánh chỉ số](figures/metric_comparison.png)

## 3. Ma trận nhầm lẫn

**Hàng = nhãn thực; cột = nhãn dự đoán**, theo thứ tự nhãn `[0, 1]`:

| Nhãn thực / Nhãn dự đoán | 0 – Đáng tin cậy | 1 – Không đáng tin cậy |
|---|---|---|
| 0 – Đáng tin cậy | TN | FP |
| 1 – Không đáng tin cậy | FN | TP |

FP là mẫu nhãn 0 bị báo thành nhãn 1; FN là mẫu nhãn 1 bị bỏ sót thành nhãn 0. Ma trận chuẩn hóa chia mỗi ô cho tổng hàng, nên mỗi hàng cộng thành 100%.

| Cấu hình SVM | TN | FP | FN | TP | Tổng lỗi |
|---|---:|---:|---:|---:|---:|
{chr(10).join(matrix_rows)}

![Tổng quan ma trận nhầm lẫn](figures/confusion_overview.png)

![Phân bố dự đoán đúng và các loại lỗi](figures/error_breakdown.png)

![Nhãn thực và nhãn dự đoán](figures/predicted_label_distribution.png)

## 4. Đối chiếu với Cross Validation

| Cấu hình SVM | F1 CV | F1 test | Test − CV (điểm %) |
|---|---:|---:|---:|
{chr(10).join(cv_rows)}

CV là trung bình các folds trên train, còn test là một tập độc lập sau kiểm tra trùng lặp. Các chênh lệch là thống kê mô tả; không xếp hạng lại hoặc chọn cấu hình dựa trên test.

![F1 CV và test](figures/cv_vs_test_f1.png)

## 5. Báo cáo chi tiết từng cấu hình

{chr(10).join(details)}

## 6. Cách tính các chỉ số

- Accuracy = (TN + TP) / (TN + FP + FN + TP).
- Precision nhãn 1 = TP / (TP + FP).
- Recall nhãn 1 = TP / (TP + FN).
- F1 nhãn 1 = 2 × Precision × Recall / (Precision + Recall).
- F1 macro là trung bình F1 hai nhãn; F1 weighted có trọng số theo số mẫu từng nhãn.
- Balanced accuracy là trung bình Recall hai nhãn. Khi mẫu số bằng 0, Precision / Recall / F1 được quy ước là 0.

Nhãn ReINTEL mô tả đáng tin cậy / không đáng tin cậy của bài đăng. Kết quả này phản ánh khả năng phân loại theo nhãn của bộ dữ liệu.

## 7. Tệp đầu ra và tái lập

{csv_links}
- [Kết quả đầy đủ và thông tin tái lập](evaluation_summary.json).
- [Báo cáo HTML ngoại tuyến](index.html).
- `figures/`: mỗi biểu đồ có bản PNG và SVG; tên tệp ma trận chứa ID cấu hình.
- `predictions/`: dự đoán từng mẫu và các mẫu sai cho từng cấu hình.

CSV ghi UTF-8 có BOM để mở trong Excel. PNG phù hợp cho báo cáo và slide; SVG giữ độ nét khi phóng to.

Chạy lại từ thư mục gốc dự án:

```powershell
.\\.venv\\Scripts\\python.exe run_evaluation.py
```

Phiên bản thư viện: {json.dumps(summary['versions'], ensure_ascii=False)}.
"""
    (docs_dir / "README.md").write_text(content, encoding="utf-8")


def render_evaluation_docs(summary: dict, docs_dir: Path) -> None:
    """Write PNG/SVG figures, README.md and self-contained index.html.

    All numerical displays derive from ``summary``. The model order remains the
    evaluation order, and the selected model is taken from ``selected_model_id``
    supplied by training provenance rather than inferred from test metrics.
    """
    docs_dir = Path(docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = docs_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "text.color": _INK,
        "axes.labelcolor": _MUTED,
        "axes.titlecolor": _INK,
        "xtick.color": _MUTED,
        "ytick.color": _MUTED,
        "svg.fonttype": "none",
    }):
        _draw_confusion_matrices(summary, figure_dir)
        _draw_metrics(summary, figure_dir)
        _draw_errors(summary, figure_dir)
        _draw_distribution(summary, figure_dir)
        _draw_cv_test(summary, figure_dir)
        _draw_flow(summary, figure_dir)
    _write_html(summary, docs_dir)
    _write_readme(summary, docs_dir)
