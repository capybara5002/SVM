"""Danh gia cac SVM da co tren test doc lap; khong fit hoac chon lai mo hinh."""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from . import config
from .load_data import DataConfigurationError


MODEL_SPECS = (
    ("svm_linear_baseline", "Linear cơ sở", "svm_linear_baseline.pkl", "linear", "baseline"),
    ("svm_rbf_baseline", "RBF cơ sở", "svm_rbf_baseline.pkl", "rbf", "baseline"),
    ("svm_linear_tuned", "Linear tối ưu", "svm_linear_best.pkl", "linear", "tuned"),
    ("svm_rbf_tuned", "RBF tối ưu", "svm_rbf_best.pkl", "rbf", "tuned"),
)
LABEL_NAMES = {0: "reliable", 1: "unreliable"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_processed_data(path: Path, split_name: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Không có {path}. Hãy chạy run_preparation.py trước.")
    data = pd.read_csv(path, encoding="utf-8-sig")
    required = {"source_index", "processed_text", "label"}
    if not required.issubset(data.columns):
        raise DataConfigurationError(f"{split_name} phải có các cột {sorted(required)}.")
    if data.empty or data[list(required)].isna().any().any():
        raise DataConfigurationError(f"{split_name} rỗng hoặc có giá trị thiếu.")
    if data["source_index"].duplicated().any():
        raise DataConfigurationError(f"source_index trong {split_name} phải duy nhất.")
    if not data["processed_text"].map(lambda value: isinstance(value, str)).all():
        raise DataConfigurationError(f"processed_text trong {split_name} phải là văn bản.")
    if data["processed_text"].str.strip().eq("").any():
        raise DataConfigurationError(f"{split_name} có văn bản rỗng.")
    if set(data["label"].unique()) != {0, 1}:
        raise DataConfigurationError(f"{split_name} phải có đúng hai nhãn 0 và 1.")
    data["label"] = data["label"].astype(int)
    return data.reset_index(drop=True)


def _label_counts(data: pd.DataFrame) -> dict[str, int]:
    return {str(label): int((data["label"] == label).sum()) for label in (0, 1)}


def prepare_test_set(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Loc theo van ban truoc khi du doan, khong thay doi cac CSV goc."""
    if set(train["source_index"]) & set(test["source_index"]):
        raise DataConfigurationError("Train và test trùng source_index; cần kiểm tra lại phép chia.")
    label_counts = test.groupby("processed_text")["label"].nunique()
    if (label_counts > 1).any():
        raise DataConfigurationError("Test có cùng văn bản nhưng nhãn mâu thuẫn.")
    matching_sources = train.groupby("processed_text")["source_index"].agg(
        lambda values: ";".join(str(value) for value in sorted(values))
    )
    overlap = test["processed_text"].isin(matching_sources.index)
    candidates = test.loc[~overlap].copy()
    duplicates = candidates["processed_text"].duplicated(keep="first")
    clean = candidates.loc[~duplicates].copy().reset_index(drop=True)
    if clean.empty or set(clean["label"].unique()) != {0, 1}:
        raise DataConfigurationError("Test sau lọc cần còn mẫu của cả hai nhãn 0 và 1.")
    overlap_rows = test.loc[overlap].copy()
    overlap_rows["exclusion_reason"] = "train_text_overlap"
    overlap_rows["matching_train_source_indices"] = overlap_rows["processed_text"].map(matching_sources)
    duplicate_rows = candidates.loc[duplicates].copy()
    duplicate_rows["exclusion_reason"] = "duplicate_test_text"
    duplicate_rows["matching_train_source_indices"] = ""
    excluded = pd.concat([overlap_rows, duplicate_rows], ignore_index=True)
    train_conflicts = train.groupby("processed_text")["label"].nunique()
    audit = {
        "train_rows": len(train),
        "original_test_rows": len(test),
        "evaluated_test_rows": len(clean),
        "excluded_rows": len(excluded),
        "excluded_overlap_rows": int(overlap.sum()),
        "excluded_duplicate_rows": int(duplicates.sum()),
        "source_index_overlap_rows": 0,
        "original_test_labels": _label_counts(test),
        "evaluated_test_labels": _label_counts(clean),
        "excluded_test_labels": _label_counts(excluded),
        "train_text_conflict_groups": int((train_conflicts > 1).sum()),
        "exclusion_policy": "Loại văn bản test trùng train, rồi giữ lần đầu của văn bản test trùng nội bộ; không dùng dự đoán hoặc điểm số để lọc.",
        "excluded_source_indices": excluded["source_index"].tolist(),
        "original_csvs_preserved": True,
    }
    return clean, excluded, audit


def compute_metrics(y_true, y_pred) -> dict:
    """Hang la nhan that, cot la nhan du doan; nhan duong la 1."""
    truth, predictions = np.asarray(y_true), np.asarray(y_pred)
    if truth.ndim != 1 or predictions.shape != truth.shape or truth.size == 0:
        raise ValueError("Nhãn thật/dự đoán phải là hai vector cùng độ dài và không rỗng.")
    if not set(np.unique(truth)).issubset({0, 1}) or not set(np.unique(predictions)).issubset({0, 1}):
        raise ValueError("Chỉ chấp nhận nhãn 0 và 1.")
    return {
        "metrics": {
            "accuracy": float(accuracy_score(truth, predictions)),
            "precision": float(precision_score(truth, predictions, pos_label=1, zero_division=0)),
            "recall": float(recall_score(truth, predictions, pos_label=1, zero_division=0)),
            "f1": float(f1_score(truth, predictions, pos_label=1, zero_division=0)),
            "f1_macro": float(f1_score(truth, predictions, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(truth, predictions, average="weighted", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        },
        "confusion_matrix": confusion_matrix(truth, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            truth, predictions, labels=[0, 1], target_names=["reliable", "unreliable"],
            output_dict=True, zero_division=0,
        ),
    }


def _prediction_table(data: pd.DataFrame, model, model_id: str) -> pd.DataFrame:
    result = data[["source_index", "processed_text", "label"]].rename(columns={"label": "y_true"}).copy()
    result.insert(0, "model_id", model_id)
    predicted = np.asarray(model.predict(data["processed_text"]))
    scores = np.asarray(model.decision_function(data["processed_text"]))
    if scores.shape != (len(data),) or not np.isfinite(scores).all():
        raise DataConfigurationError(f"Điểm quyết định không hợp lệ: {model_id}.")
    if predicted.shape != (len(data),) or not set(np.unique(predicted)).issubset({0, 1}):
        raise DataConfigurationError(f"Nhãn dự đoán không hợp lệ: {model_id}.")
    result["y_pred"] = predicted.astype(int)
    result["decision_score"] = scores
    result["true_label_name"] = result["y_true"].map(LABEL_NAMES)
    result["predicted_label_name"] = result["y_pred"].map(LABEL_NAMES)
    result["is_correct"] = result["y_true"].eq(result["y_pred"])
    result["outcome"] = np.select(
        [
            result["y_true"].eq(0) & result["y_pred"].eq(0),
            result["y_true"].eq(0) & result["y_pred"].eq(1),
            result["y_true"].eq(1) & result["y_pred"].eq(0),
            result["y_true"].eq(1) & result["y_pred"].eq(1),
        ],
        ["TN", "FP", "FN", "TP"], default="invalid",
    )
    return result


def _validate_model(model, params: dict, train_rows: int, name: str) -> None:
    if not hasattr(model, "named_steps") or not {"tfidf", "svc"}.issubset(model.named_steps):
        raise DataConfigurationError(f"{name} phải chứa pipeline TF-IDF và SVC.")
    svc = model.named_steps["svc"]
    if list(svc.classes_) != [0, 1] or svc.fit_status_ != 0 or svc.shape_fit_[0] != train_rows:
        raise DataConfigurationError(f"{name} không khớp nhãn hoặc số mẫu train trong metadata.")
    for parameter in ("kernel", "C", "gamma", "class_weight"):
        if getattr(svc, parameter) != params[parameter]:
            raise DataConfigurationError(f"{name}: tham số {parameter} khác metadata Tuần 5.")


def _write_csv(data: pd.DataFrame, path: Path) -> None:
    data.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.12g")


def evaluate_saved_models(
    train_path: Path = config.PROCESSED_DATA_DIR / "train.csv",
    test_path: Path = config.PROCESSED_DATA_DIR / "test.csv",
    training_summary_path: Path = config.TRAINING_RESULTS_DIR / "training_summary.json",
    artifacts_dir: Path = config.ARTIFACTS_DIR,
    docs_dir: Path = config.DOCS_DIR,
    report_path: Path = config.RESULTS_DIR / "bao_cao_tien_do_tuan_6.md",
) -> dict:
    started = perf_counter()
    metadata = json.loads(training_summary_path.read_text(encoding="utf-8"))
    train_hash, test_hash = _sha256(train_path), _sha256(test_path)
    if metadata["train_sha256"] != train_hash:
        raise DataConfigurationError("Train đã thay đổi so với lần huấn luyện; không thể xác nhận mô hình phù hợp.")
    train = load_processed_data(train_path, "Train")
    test = load_processed_data(test_path, "Test")
    clean, excluded, audit = prepare_test_set(train, test)
    chosen_kernel = metadata["best_model"]["kernel"]
    if chosen_kernel not in {"linear", "rbf"}:
        raise DataConfigurationError("Kernel đã chọn trong metadata không được hỗ trợ.")
    selected_id = f"svm_{chosen_kernel}_tuned"
    selected_file = artifacts_dir / f"svm_{chosen_kernel}_best.pkl"
    alias_file = artifacts_dir / "svm_best_model.pkl"
    if _sha256(alias_file) != _sha256(selected_file):
        raise DataConfigurationError("svm_best_model.pkl không khớp mô hình đã chọn ở Tuần 5.")

    # Nap va kiem tra tat ca mo hinh truoc khi xem ket qua test.
    frozen_models = []
    for model_id, display_name, filename, kernel, stage in MODEL_SPECS:
        record = metadata["baselines"][model_id] if stage == "baseline" else metadata["searches"][kernel]
        params = (
            dict(record["params"]) if stage == "baseline" else
            {"kernel": kernel, "C": record["best_params"]["svc__C"],
             "gamma": record["best_params"].get("svc__gamma", "scale"),
             "class_weight": record["best_params"]["svc__class_weight"]}
        )
        model_file = artifacts_dir / filename
        model_hash = _sha256(model_file)
        model = joblib.load(model_file)
        _validate_model(model, params, len(train), filename)
        frozen_models.append((model_id, display_name, model_file, model_hash, model, params, record))

    tables_dir, predictions_dir = docs_dir / "tables", docs_dir / "predictions"
    for directory in (tables_dir, predictions_dir, report_path.parent):
        directory.mkdir(parents=True, exist_ok=True)
    _write_csv(excluded, tables_dir / "test_data_audit.csv")
    _write_csv(clean, tables_dir / "evaluated_test_samples.csv")
    conflicting_texts = train.groupby("processed_text")["label"].nunique()
    conflicts = train[train["processed_text"].isin(conflicting_texts[conflicting_texts > 1].index)]
    _write_csv(conflicts, tables_dir / "train_label_conflicts.csv")
    summary = {
        "completed_at": None,
        "data_audit": {**audit, "train_sha256": train_hash, "test_sha256": test_hash},
        "selected_model_id": selected_id,
        "model_selection": {
            "source": "week4_5_cv", "kernel": chosen_kernel,
            "params": metadata["searches"][chosen_kernel]["best_params"],
            "mean_cv_f1": metadata["best_model"]["mean_cv_f1"],
            "threshold": "Ngưỡng dự đoán mặc định của SVC (decision_function = 0)",
            "alias_file": str(alias_file.resolve()), "alias_sha256": _sha256(alias_file),
            "test_used_for_selection": False, "refit_during_evaluation": False,
        },
        "models": [],
        "tables": {
            "metrics": "tables/model_metrics.csv",
            "classification_report": "tables/classification_report.csv",
            "confusion_matrices": "tables/confusion_matrices.csv",
            "data_audit": "tables/test_data_audit.csv",
            "evaluated_test_samples": "tables/evaluated_test_samples.csv",
            "train_label_conflicts": "tables/train_label_conflicts.csv",
            "predictions": "predictions/all_predictions.csv",
        },
        "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__},
    }
    metric_rows, class_rows, matrix_rows, prediction_frames = [], [], [], []
    print(f"Test gốc: {len(test)} | Loại trùng: {len(excluded)} | Đánh giá: {len(clean)} mẫu", flush=True)
    for model_id, display_name, model_file, model_hash, model, params, record in frozen_models:
        before = perf_counter()
        predictions = _prediction_table(clean, model, model_id)
        scores = compute_metrics(predictions["y_true"], predictions["y_pred"])
        prediction_file = f"predictions/{model_id}.csv"
        error_file = f"predictions/{model_id}_errors.csv"
        _write_csv(predictions, docs_dir / prediction_file)
        _write_csv(predictions.loc[~predictions["is_correct"]], docs_dir / error_file)
        model_result = {
            "id": model_id, "display_name": display_name, "params": params, **scores,
            "cv_f1": record["metrics"]["f1"]["mean"],
            "model_file": str(model_file.resolve()), "model_sha256": model_hash,
            "prediction_file": prediction_file, "error_file": error_file,
            "evaluated_samples": len(clean), "seconds": perf_counter() - before,
        }
        summary["models"].append(model_result)
        tn, fp, fn, tp = np.asarray(scores["confusion_matrix"]).ravel().tolist()
        metric_rows.append({
            "model_id": model_id, "display_name": display_name,
            "selected_by_cv": model_id == selected_id, "params": json.dumps(params),
            "samples": len(clean), "cv_f1": model_result["cv_f1"], **scores["metrics"],
            "tn": tn, "fp": fp, "fn": fn, "tp": tp, "errors": fp + fn,
        })
        for class_name, values in scores["classification_report"].items():
            if isinstance(values, dict):
                class_rows.append({"model_id": model_id, "class": class_name, **values})
        for true_label in (0, 1):
            row_total = sum(scores["confusion_matrix"][true_label])
            for predicted_label in (0, 1):
                count = scores["confusion_matrix"][true_label][predicted_label]
                matrix_rows.append({
                    "model_id": model_id, "true_label": true_label, "predicted_label": predicted_label,
                    "count": count, "row_fraction": count / row_total,
                })
        prediction_frames.append(predictions)
        print(f"  {display_name}: Accuracy={scores['metrics']['accuracy']:.4f}, F1={scores['metrics']['f1']:.4f}", flush=True)
        if _sha256(model_file) != model_hash:
            raise RuntimeError(f"Tệp mô hình đã thay đổi trong khi đánh giá: {model_file}.")

    _write_csv(pd.DataFrame(metric_rows), tables_dir / "model_metrics.csv")
    _write_csv(pd.DataFrame(class_rows), tables_dir / "classification_report.csv")
    _write_csv(pd.DataFrame(matrix_rows), tables_dir / "confusion_matrices.csv")
    _write_csv(pd.concat(prediction_frames, ignore_index=True), predictions_dir / "all_predictions.csv")
    if _sha256(train_path) != train_hash or _sha256(test_path) != test_hash:
        raise RuntimeError("CSV gốc đã thay đổi trong khi đánh giá.")
    summary["completed_at"] = datetime.now(timezone(timedelta(hours=7))).isoformat(timespec="seconds")
    summary["total_seconds"] = perf_counter() - started
    (docs_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    from .evaluation_visuals import render_evaluation_docs

    render_evaluation_docs(summary, docs_dir)
    chosen = next(model for model in summary["models"] if model["id"] == selected_id)
    metrics = chosen["metrics"]
    report_path.write_text(
        "# Báo cáo tiến độ Tuần 6\n\n"
        f"Hoàn tất đánh giá lúc {summary['completed_at']} (UTC+7).\n\n"
        f"Test gốc {len(test)} mẫu; loại {len(excluded)} mẫu theo quy tắc kiểm tra trùng, còn {len(clean)} mẫu. "
        "Đã đánh giá bốn cấu hình SVM cố định, xuất dự đoán, bảng chỉ số, báo cáo theo nhãn và biểu đồ.\n\n"
        f"Mô hình đã chọn từ CV: **{chosen['display_name']}**, {chosen['params']}.\n\n"
        "| Accuracy | Precision nhãn 1 | Recall nhãn 1 | F1 nhãn 1 |\n|---:|---:|---:|---:|\n"
        f"| {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} |\n\n"
        "[Xem báo cáo đầy đủ, bảng và biểu đồ](../docs/README.md). "
        "[Mở trang tổng hợp](../docs/index.html).\n\n"
        "Không huấn luyện lại hoặc lựa chọn lại tham số từ kết quả test. Tuần 7–8 sẽ tổng hợp báo cáo đồ án.\n",
        encoding="utf-8",
    )
    print(f"Hoàn tất Tuần 6. Xem {docs_dir / 'index.html'}", flush=True)
    return summary
