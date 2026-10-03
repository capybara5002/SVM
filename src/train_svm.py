"""Huan luyen SVM tuan 4-5, chi su dung train va TF-IDF trong tung fold."""

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
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from . import config
from .load_data import DataConfigurationError
from .training_report import write_training_report
from .vectorize import create_tfidf_vectorizer


SCORING = {
    "f1": make_scorer(f1_score, pos_label=1, zero_division=0),
    "precision": make_scorer(precision_score, pos_label=1, zero_division=0),
    "recall": make_scorer(recall_score, pos_label=1, zero_division=0),
    "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
}


def load_training_data(path: Path, n_splits: int = config.CV_FOLDS) -> pd.DataFrame:
    """Kiem tra dau vao; khong doc test.csv hoac ma tran fit san tu tuan 3."""
    if not path.is_file():
        raise FileNotFoundError(f"Không có {path}. Hãy chạy run_preparation.py trước.")
    data = pd.read_csv(path, encoding="utf-8-sig")
    required = {"source_index", "processed_text", "label"}
    if not required.issubset(data.columns):
        raise DataConfigurationError(f"Train phải có các cột: {sorted(required)}.")
    if data[list(required)].isna().any().any():
        raise DataConfigurationError("Train có giá trị thiếu trong cột bắt buộc.")
    if data["source_index"].duplicated().any():
        raise DataConfigurationError("source_index trong train phải duy nhất.")
    if not data["processed_text"].map(lambda value: isinstance(value, str)).all():
        raise DataConfigurationError("processed_text phải là văn bản.")
    if data["processed_text"].str.strip().eq("").any():
        raise DataConfigurationError("Train có văn bản rỗng sau tiền xử lý.")
    if set(data["label"].unique()) != {0, 1}:
        raise DataConfigurationError("Nhãn phải đúng 0 = reliable và 1 = unreliable.")
    if data["label"].value_counts().min() < n_splits:
        raise DataConfigurationError(f"Mỗi nhãn cần ít nhất {n_splits} mẫu để chia CV.")
    data["label"] = data["label"].astype(int)
    return data.reset_index(drop=True)


def make_cv_splits(
    data: pd.DataFrame, n_splits: int = config.CV_FOLDS
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    """Giu cac van ban trung nhau cung fold neu can; moi fold co ca hai nhan."""
    texts, labels = data["processed_text"], data["label"]
    if texts.duplicated().any():
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE
        )
        splits = list(splitter.split(texts, labels, groups=texts))
    else:
        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE
        )
        splits = list(splitter.split(texts, labels))
    for train_indices, validation_indices in splits:
        if any(labels.iloc[indices].nunique() != 2 for indices in (train_indices, validation_indices)):
            raise DataConfigurationError("Không chia được CV có cả hai nhãn trong mỗi fold.")
        if set(texts.iloc[train_indices]) & set(texts.iloc[validation_indices]):
            raise DataConfigurationError("Có văn bản trùng giữa train và validation của một fold.")
    return splits, type(splitter).__name__


def build_pipeline(kernel: str = "linear") -> Pipeline:
    """TF-IDF chua fit, duoc clone va hoc rieng trong moi lan fit cua CV."""
    if kernel not in {"linear", "rbf"}:
        raise ValueError("Chỉ hỗ trợ kernel linear hoặc rbf.")
    return Pipeline(
        [
            ("tfidf", create_tfidf_vectorizer()),
            ("svc", SVC(kernel=kernel, C=1.0, gamma="scale", cache_size=256)),
        ]
    )


def parameter_grid(kernel: str) -> dict[str, list]:
    if kernel not in {"linear", "rbf"}:
        raise ValueError("Chỉ hỗ trợ kernel linear hoặc rbf.")
    grid = {
        "svc__C": list(config.SVM_C_VALUES),
        "svc__class_weight": list(config.SVM_CLASS_WEIGHTS),
    }
    if kernel == "rbf":
        grid["svc__gamma"] = list(config.SVM_GAMMA_VALUES)
    return grid


def build_grid_search(kernel: str, cv: list, n_jobs: int, verbose: int = 1) -> GridSearchCV:
    return GridSearchCV(
        build_pipeline(kernel), parameter_grid(kernel), scoring=SCORING,
        refit="f1", cv=cv, n_jobs=n_jobs, pre_dispatch=n_jobs,
        return_train_score=False, error_score="raise", verbose=verbose,
    )


def _metric_summary(fold_values: dict[str, list[float]]) -> dict:
    return {
        name: {"mean": float(np.mean(values)), "std": float(np.std(values)), "folds": values}
        for name, values in fold_values.items()
    }


def _save_model(model: Pipeline, path: Path, sample_texts: pd.Series) -> None:
    """Luu ca vectorizer va SVM, kiem tra du doan sau nap lai tren mau train."""
    joblib.dump(model, path, compress=3)
    restored = joblib.load(path)
    if not np.array_equal(model.predict(sample_texts), restored.predict(sample_texts)):
        raise RuntimeError(f"Dự đoán thay đổi sau khi nạp lại mô hình: {path}.")


def train_models(
    train_path: Path = config.PROCESSED_DATA_DIR / "train.csv",
    results_dir: Path = config.TRAINING_RESULTS_DIR,
    artifacts_dir: Path = config.ARTIFACTS_DIR,
    report_path: Path = config.RESULTS_DIR / "bao_cao_tien_do_tuan_5.md",
    n_jobs: int = 2,
    verbose: int = 1,
) -> dict:
    """Baseline, Grid Search Linear/RBF va refit mo hinh tot nhat tren train."""
    if n_jobs < 1:
        raise ValueError("n_jobs phải là số nguyên dương.")
    started = perf_counter()
    data = load_training_data(train_path)
    texts, labels = data["processed_text"], data["label"]
    cv, cv_type = make_cv_splits(data)
    for directory in (results_dir, artifacts_dir, report_path.parent):
        directory.mkdir(parents=True, exist_ok=True)

    assignments = data[["source_index", "label"]].copy()
    assignments["validation_fold"] = -1
    fold_details = []
    for fold, (fit_indices, validation_indices) in enumerate(cv, start=1):
        assignments.loc[validation_indices, "validation_fold"] = fold
        fold_details.append({
            "fold": fold, "fit_rows": len(fit_indices), "validation_rows": len(validation_indices),
            "fit_labels": {str(k): int(v) for k, v in labels.iloc[fit_indices].value_counts().items()},
            "validation_labels": {str(k): int(v) for k, v in labels.iloc[validation_indices].value_counts().items()},
        })
    assignments.to_csv(results_dir / "cv_fold_assignments.csv", index=False, encoding="utf-8-sig")

    print(f"Train: {len(data)} mẫu | {cv_type}: {len(cv)} folds | n_jobs={n_jobs}", flush=True)
    summary = {
        "completed_at": None,
        "train_file": str(train_path.resolve()),
        "train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "train_rows": len(data),
        "train_labels": {str(k): int(v) for k, v in labels.value_counts().sort_index().items()},
        "duplicate_processed_text_rows": int(texts.duplicated().sum()),
        "cv": {"type": cv_type, "folds": config.CV_FOLDS, "random_state": config.RANDOM_STATE, "details": fold_details},
        "selection_metric": "f1", "positive_label": 1,
        "test_used": False,
        "tfidf": {"max_features": config.MAX_FEATURES, "ngram_range": list(config.NGRAM_RANGE), "fit_inside_cv": True},
        "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__},
        "n_jobs": n_jobs, "baselines": {}, "searches": {},
    }
    comparison_rows, baseline_fold_rows, best_fold_rows = [], [], []

    print("[1/3] Kiểm định mô hình cơ sở (Dummy, SVM Linear và SVM RBF)", flush=True)
    baselines = {
        "dummy_majority": Pipeline([("tfidf", create_tfidf_vectorizer()), ("dummy", DummyClassifier(strategy="most_frequent"))]),
        "svm_linear_baseline": build_pipeline("linear"),
        "svm_rbf_baseline": build_pipeline("rbf"),
    }
    for name, model in baselines.items():
        before = perf_counter()
        scores = cross_validate(model, texts, labels, cv=cv, scoring=SCORING, n_jobs=n_jobs, pre_dispatch=n_jobs, error_score="raise")
        metrics = _metric_summary({metric: scores[f"test_{metric}"].tolist() for metric in SCORING})
        model_path = None
        if name != "dummy_majority":
            model.fit(texts, labels)
            model_path = artifacts_dir / f"{name}.pkl"
            _save_model(model, model_path, texts.head(10))
        params = {"strategy": "most_frequent"} if name == "dummy_majority" else {"C": 1.0, "kernel": model.named_steps["svc"].kernel, "gamma": "scale", "class_weight": None}
        summary["baselines"][name] = {"params": params, "metrics": metrics, "seconds": perf_counter() - before, "model_file": str(model_path.resolve()) if model_path else None}
        comparison_rows.append({"model": name, "stage": "baseline", "params": json.dumps(params), **{f"mean_cv_{k}": v["mean"] for k, v in metrics.items()}, "std_cv_f1": metrics["f1"]["std"]})
        for fold in range(len(cv)):
            baseline_fold_rows.append({"model": name, "fold": fold + 1, **{metric: metrics[metric]["folds"][fold] for metric in SCORING}})
        print(f"  {name}: F1={metrics['f1']['mean']:.4f} | Accuracy={metrics['accuracy']['mean']:.4f}", flush=True)
    pd.DataFrame(baseline_fold_rows).to_csv(results_dir / "baseline_cv_folds.csv", index=False, encoding="utf-8-sig")

    print("[2/3] Grid Search chọn C, gamma và class_weight", flush=True)
    winner_model, winner_kernel, winner_score = None, None, -1.0
    for kernel in ("linear", "rbf"):
        before = perf_counter()
        search = build_grid_search(kernel, cv, n_jobs, verbose)
        candidate_count = len(ParameterGrid(search.param_grid))
        print(f"  {kernel}: {candidate_count} cấu hình × {len(cv)} folds", flush=True)
        search.fit(texts, labels)
        cv_results = pd.DataFrame(search.cv_results_)
        cv_results["params"] = cv_results["params"].map(lambda value: json.dumps(value, sort_keys=True))
        cv_results.to_csv(results_dir / f"grid_search_{kernel}.csv", index=False, encoding="utf-8-sig")
        index = search.best_index_
        metrics = _metric_summary({metric: [float(search.cv_results_[f"split{fold}_test_{metric}"][index]) for fold in range(len(cv))] for metric in SCORING})
        model = search.best_estimator_
        model_path = artifacts_dir / f"svm_{kernel}_best.pkl"
        _save_model(model, model_path, texts.head(10))
        svc = model.named_steps["svc"]
        summary["searches"][kernel] = {
            "parameter_grid": search.param_grid, "candidates": candidate_count,
            "cv_fits": candidate_count * len(cv), "best_params": search.best_params_,
            "best_candidate_index": int(index), "metrics": metrics,
            "support_vectors_by_class": svc.n_support_.tolist(),
            "fitted_features": len(model.named_steps["tfidf"].vocabulary_),
            "seconds": perf_counter() - before, "model_file": str(model_path.resolve()),
        }
        comparison_rows.append({"model": f"svm_{kernel}_tuned", "stage": "grid_search", "params": json.dumps(search.best_params_), **{f"mean_cv_{k}": v["mean"] for k, v in metrics.items()}, "std_cv_f1": metrics["f1"]["std"]})
        for fold in range(len(cv)):
            best_fold_rows.append({"model": f"svm_{kernel}_tuned", "fold": fold + 1, **{metric: metrics[metric]["folds"][fold] for metric in SCORING}})
        if search.best_score_ > winner_score:
            winner_model, winner_kernel, winner_score = model, kernel, float(search.best_score_)
        print(f"  {kernel}: tốt nhất F1={search.best_score_:.4f}, {search.best_params_}", flush=True)

    print("[3/3] Lưu mô hình tốt nhất và báo cáo Tuần 5", flush=True)
    winner_path = artifacts_dir / "svm_best_model.pkl"
    _save_model(winner_model, winner_path, texts.head(10))
    summary["best_model"] = {"kernel": winner_kernel, "mean_cv_f1": winner_score, "model_file": str(winner_path.resolve()), "input": "processed_text: văn bản đã clean_text và tách từ bằng Underthesea"}
    summary["total_seconds"] = perf_counter() - started
    summary["completed_at"] = datetime.now(timezone(timedelta(hours=7))).isoformat(timespec="seconds")
    pd.DataFrame(comparison_rows).to_csv(results_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(best_fold_rows).to_csv(results_dir / "best_cv_folds.csv", index=False, encoding="utf-8-sig")
    (results_dir / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    write_training_report(summary, report_path)
    print(f"Hoàn tất: {winner_kernel}, F1 CV={winner_score:.4f}. Mô hình: {winner_path}", flush=True)
    return summary
