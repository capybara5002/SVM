"""Controlled 2x2 SVM experiment; frozen source IDs and fold-local TF-IDF."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import NearestNeighbors

from . import config
from .evaluate_svm import compute_metrics, load_processed_data, prepare_test_set
from .preprocess import clean_text, segment_vietnamese
from .train_svm import SCORING, build_pipeline, parameter_grid

OUT = config.RESULTS_DIR / "chapter3_4"
SPECS = (("A", False, "linear"), ("B", False, "rbf"),
         ("C", True, "linear"), ("D", True, "rbf"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(value, path: Path) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.14g")


def audit_inputs(train: pd.DataFrame, test: pd.DataFrame, raw: pd.DataFrame,
                 assignments: pd.DataFrame, expected_counts=(3428, 856)) -> tuple[pd.DataFrame, pd.DataFrame, list, dict]:
    """Reject changed identities, labels, invalid folds or clean-text leakage."""
    if (len(train), len(test)) != expected_counts:
        raise ValueError("Expected the frozen 3428 train / 856 filtered holdout IDs")
    if assignments.source_index.duplicated().any() or set(assignments.source_index) != set(train.source_index):
        raise ValueError("Fold IDs do not match train IDs one-to-one")
    if set(assignments.validation_fold) != set(range(1, 6)):
        raise ValueError("Expected exactly folds 1..5")
    if set(train.source_index) & set(test.source_index):
        raise ValueError("Overlapping source IDs")
    for frame in (train, test):
        if frame.source_index.duplicated().any() or not frame.source_index.isin(raw.index).all():
            raise ValueError("Invalid or duplicate source ID")
        if not np.array_equal(frame.label.to_numpy(), raw.loc[frame.source_index, "label"].to_numpy()):
            raise ValueError("Labels disagree with raw source")
    if not np.array_equal(assignments.set_index("source_index").loc[train.source_index, "label"], train.label):
        raise ValueError("Fold labels disagree with train")
    train, test = train.copy(), test.copy()
    for frame in (train, test):
        frame["text_original"] = raw.loc[frame.source_index, "text"].to_numpy()
        frame["text_clean"] = frame.text_original.map(clean_text)
        if frame.text_clean.eq("").any():
            raise ValueError("Empty cleaned text")
    if set(train.text_clean) & set(test.text_clean) or test.text_clean.duplicated().any():
        raise ValueError("Exact cleaned holdout leakage/duplicates")
    train["validation_fold"] = assignments.set_index("source_index").loc[train.source_index, "validation_fold"].to_numpy()
    folds, details = [], []
    for fold in range(1, 6):
        fit = np.flatnonzero(train.validation_fold.ne(fold))
        valid = np.flatnonzero(train.validation_fold.eq(fold))
        for key in ("text_clean", "processed_text"):
            if set(train.iloc[fit][key]) & set(train.iloc[valid][key]):
                raise ValueError(f"Fold {fold}: {key} leakage")
        for indices in (fit, valid):
            if set(train.iloc[indices].label) != {0, 1}:
                raise ValueError("Each fold partition needs both labels")
        folds.append((fit, valid))
        details.append({"fold": fold, "train_n": len(fit), "validation_n": len(valid),
                        "train_0": int(train.iloc[fit].label.eq(0).sum()),
                        "train_1": int(train.iloc[fit].label.eq(1).sum()),
                        "validation_0": int(train.iloc[valid].label.eq(0).sum()),
                        "validation_1": int(train.iloc[valid].label.eq(1).sum()),
                        "clean_text_overlap": 0, "segmented_text_overlap": 0})
    conflicts = train.groupby("text_clean").label.nunique()
    return train, test, folds, {"train_rows": len(train), "test_rows": len(test),
        "test_labels": {str(k): int(v) for k, v in test.label.value_counts().sort_index().items()},
        "source_id_overlap": 0, "clean_text_overlap": 0,
        "train_duplicate_clean_rows": int(train.text_clean.duplicated().sum()),
        "test_duplicate_clean_rows": 0, "train_conflicting_clean_groups": int(conflicts.gt(1).sum()),
        "folds_reused": True, "fold_details": details}


def benchmark(model, raw_texts: list[str], segmented: bool, repeats: int = 5, batch_size: int = 64) -> dict:
    """Warm CPU benchmark; identical 128 IDs, batches and repeats for all scenarios."""
    processed = [segment_vietnamese(x) if segmented else clean_text(x) for x in raw_texts]
    model.predict(processed[:batch_size])
    timings, preprocessing, end_to_end = [], [], []
    for _ in range(repeats):
        start = perf_counter()
        for offset in range(0, len(processed), batch_size):
            model.predict(processed[offset:offset + batch_size])
        timings.append(perf_counter() - start)
        start = perf_counter()
        texts = [segment_vietnamese(x) if segmented else clean_text(x) for x in raw_texts]
        preprocess_seconds = perf_counter() - start
        for offset in range(0, len(texts), batch_size):
            model.predict(texts[offset:offset + batch_size])
        end_to_end.append(perf_counter() - start)
        preprocessing.append(preprocess_seconds)
    return {"samples": len(raw_texts), "batch_size": batch_size, "repeats": repeats, "warmup_batches": 1,
            "operation": "TF-IDF.transform + SVC.predict; preprocessing timed separately; no decision_function",
            "pipeline_seconds": timings, "preprocessing_seconds": preprocessing, "end_to_end_seconds": end_to_end,
            "pipeline_median_ms_per_sample": float(np.median(timings) * 1000 / len(raw_texts)),
            "pipeline_std_ms_per_sample": float(np.std(timings) * 1000 / len(raw_texts)),
            "end_to_end_median_ms_per_sample": float(np.median(end_to_end) * 1000 / len(raw_texts))}


def sensitivity(train: pd.DataFrame, test: pd.DataFrame, predictions: pd.DataFrame, output: Path) -> dict:
    """Lexical near-duplicate sensitivity; train-only character vocabulary, no retuning."""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=50000, lowercase=False)
    train_x = vec.fit_transform(train.text_clean)
    test_x = vec.transform(test.text_clean)
    distances, indices = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute", n_jobs=1).fit(train_x).kneighbors(test_x)
    pairs = pd.DataFrame({"source_index": test.source_index, "nearest_train_id": train.iloc[indices[:, 0]].source_index.to_numpy(),
                          "cosine_similarity": 1 - distances[:, 0], "label": test.label,
                          "nearest_train_label": train.iloc[indices[:, 0]].label.to_numpy()})
    write_csv(pairs, output / "near_duplicate_pairs.csv")
    rows = []
    for threshold in (0.8, 0.9, 0.95):
        ids = set(pairs.loc[pairs.cosine_similarity < threshold, "source_index"])
        for scenario, frame in predictions.groupby("scenario"):
            subset = frame.loc[frame.source_index.isin(ids)]
            metrics = compute_metrics(subset.y_true, subset.y_pred)
            rows.append({"threshold": threshold, "scenario": scenario, "retained": len(subset),
                         "excluded": len(test) - len(subset), "positive_support": int(subset.y_true.sum()),
                         **metrics["metrics"], "roc_auc": float(roc_auc_score(subset.y_true, subset.decision_score))})
    write_csv(pd.DataFrame(rows), output / "near_duplicate_sensitivity.csv")
    return {"method": "train-fitted char_wb TF-IDF (3,5), max_features=50000; cosine nearest train text",
            "thresholds": [0.8, 0.9, 0.95], "retune": False,
            "interpretation": "Exploratory lexical sensitivity; not semantic/event deduplication or a new independent test"}


def run(n_jobs: int = 2) -> dict:
    if n_jobs < 1:
        raise ValueError("n_jobs must be positive")
    OUT.mkdir(parents=True, exist_ok=True)
    model_dir = config.ARTIFACTS_DIR / "chapter3_4"
    model_dir.mkdir(parents=True, exist_ok=True)
    paths = {"raw": config.RAW_DATA_DIR / "reintel_train.csv",
             "train": config.PROCESSED_DATA_DIR / "train.csv", "test": config.PROCESSED_DATA_DIR / "test.csv",
             "filtered_test": config.DOCS_DIR / "tables/evaluated_test_samples.csv",
             "folds": config.TRAINING_RESULTS_DIR / "cv_fold_assignments.csv"}
    hashes = {k: digest(p) for k, p in paths.items()}
    raw = pd.read_csv(paths["raw"], encoding="utf-8-sig")
    train = load_processed_data(paths["train"], "train")
    original_test = load_processed_data(paths["test"], "test")
    test, _, old_audit = prepare_test_set(train, original_test)
    frozen = pd.read_csv(paths["filtered_test"], encoding="utf-8-sig")
    if not test.equals(frozen):
        raise ValueError("Filtered holdout changed from legacy evaluation")
    train, test, folds, audit = audit_inputs(train, test, raw, pd.read_csv(paths["folds"]))
    audit["original_test_rows"] = len(original_test)
    audit["excluded_test_ids"] = old_audit["excluded_source_indices"]
    write_json(audit, OUT / "data_audit.json")
    write_csv(train[["source_index", "label", "validation_fold"]], OUT / "cv_fold_assignments.csv")
    write_csv(test[["source_index", "label"]], OUT / "holdout_ids.csv")
    write_csv(pd.DataFrame(audit["fold_details"]), OUT / "fold_summary.csv")
    preprocessing = {}
    for segmented in (False, True):
        column = "underthesea" if segmented else "no_underthesea"
        preprocessing[column] = {}
        for name, frame in (("train", train), ("test", test)):
            before = perf_counter()
            frame[column] = frame.text_original.map(segment_vietnamese if segmented else clean_text)
            preprocessing[column][name + "_seconds"] = perf_counter() - before
            if segmented and not frame[column].equals(frame.processed_text):
                raise ValueError("Current Underthesea output differs from frozen processed_text")
    cpu = subprocess.check_output(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"], text=True).strip()
    summary = {"started_at": datetime.now(timezone(timedelta(hours=7))).isoformat(),
        "input_sha256": hashes, "audit": audit, "preprocessing": preprocessing,
        "hardware": {"cpu": cpu, "logical_cpus": psutil.cpu_count(), "physical_cpus": psutil.cpu_count(logical=False),
                     "ram_bytes": psutil.virtual_memory().total, "os": platform.platform()},
        "versions": {p: importlib.metadata.version(p) for p in ("scikit-learn", "numpy", "pandas", "scipy", "underthesea", "matplotlib", "joblib", "psutil")},
        "python": platform.python_version(), "n_jobs": n_jobs, "models": [],
        "protocol": {"scoring": "F1 positive_label=1; mean of 5 validation folds; std ddof=0",
            "tie_break": "first candidate in sklearn ParameterGrid order, then A/B/C/D order for global ties",
            "tfidf": {"max_features": 10000, "ngram_range": [1, 1], "token_pattern": r"(?u)\b\w+\b", "lowercase": False,
                      "norm": "l2", "use_idf": True, "smooth_idf": True, "sublinear_tf": False, "min_df": 1, "max_df": 1.0,
                      "fit_inside_each_fold": True},
            "model_memory": "Not measured; serialized artifact bytes are disk size, not RAM usage",
            "threshold": 0, "holdout_used_for_selection": False}}
    fitted = {}
    # Freeze every scenario's hyperparameters and overall CV winner before scoring holdout.
    for scenario, segmented, kernel in SPECS:
        column = "underthesea" if segmented else "no_underthesea"
        print(f"{scenario}: {column} / {kernel}: Grid Search", flush=True)
        search = GridSearchCV(build_pipeline(kernel), parameter_grid(kernel), cv=folds, scoring=SCORING,
                              refit=False, n_jobs=n_jobs, pre_dispatch=n_jobs, error_score="raise", verbose=1)
        before = perf_counter()
        search.fit(train[column], train.label)
        grid_seconds = perf_counter() - before
        results = pd.DataFrame(search.cv_results_)
        best = int(np.argmax(results.mean_test_f1.to_numpy()))
        params = search.cv_results_["params"][best]
        results["params"] = results.params.map(lambda p: json.dumps(p, sort_keys=True))
        write_csv(results, OUT / f"grid_search_{scenario}.csv")
        model = clone(build_pipeline(kernel)).set_params(**params)
        before = perf_counter()
        model.fit(train[column], train.label)
        refit_seconds = perf_counter() - before
        if list(model.classes_) != [0, 1]:
            raise ValueError("Unexpected decision score orientation")
        path = model_dir / f"svm_{scenario}.pkl"
        joblib.dump(model, path, compress=3)
        if not np.array_equal(model.predict(train[column].head(8)), joblib.load(path).predict(train[column].head(8))):
            raise ValueError("Serialized model round-trip failed")
        record = {"scenario": scenario, "underthesea": segmented, "kernel": kernel, "params": params,
            "cv_f1_mean": float(results.loc[best, "mean_test_f1"]), "cv_f1_std": float(results.loc[best, "std_test_f1"]),
            "cv_fold_f1": [float(results.loc[best, f"split{i}_test_f1"]) for i in range(5)],
            "candidate_count": len(results), "cv_fits": len(results) * 5, "grid_search_seconds": grid_seconds,
            "refit_seconds": refit_seconds, "model_file": str(path.relative_to(config.BASE_DIR)),
            "model_sha256": digest(path), "model_bytes": path.stat().st_size,
            "features": len(model.named_steps["tfidf"].vocabulary_), "support_vectors": model.named_steps["svc"].n_support_.tolist()}
        summary["models"].append(record)
        fitted[scenario] = model
        print(f"{scenario}: CV F1={record['cv_f1_mean']:.6f}, {params}", flush=True)
        write_json(summary, OUT / "training_progress.json")
    selected = max(summary["models"], key=lambda r: r["cv_f1_mean"])["scenario"]
    summary["selected_by_cv"] = selected
    write_json(summary, OUT / "selection_before_holdout.json")
    predictions, metric_rows = [], []
    for record in summary["models"]:
        scenario = record["scenario"]
        model = fitted[scenario]
        column = "underthesea" if record["underthesea"] else "no_underthesea"
        print(f"{scenario}: holdout + fixed-batch benchmark", flush=True)
        scores = model.decision_function(test[column])
        predicted = model.predict(test[column])
        result = compute_metrics(test.label, predicted)
        result["metrics"]["roc_auc"] = float(roc_auc_score(test.label, scores))
        record.update(result)
        record["benchmark"] = benchmark(model, test.text_original.head(128).tolist(), record["underthesea"])
        frame = pd.DataFrame({"scenario": scenario, "source_index": test.source_index, "y_true": test.label,
                              "y_pred": predicted, "decision_score": scores, "text_original": test.text_original})
        frame["outcome"] = np.select([(frame.y_true == 0) & (frame.y_pred == 0), (frame.y_true == 0) & (frame.y_pred == 1),
                                      (frame.y_true == 1) & (frame.y_pred == 0)], ["TN", "FP", "FN"], default="TP")
        write_csv(frame, OUT / f"predictions_{scenario}.csv")
        predictions.append(frame)
        tn, fp, fn, tp = np.asarray(result["confusion_matrix"]).ravel()
        metric_rows.append({"scenario": scenario, "underthesea": record["underthesea"], "kernel": record["kernel"],
            "selected_by_cv": scenario == selected, "params": json.dumps(record["params"]),
            "cv_f1_mean": record["cv_f1_mean"], "cv_f1_std": record["cv_f1_std"], **result["metrics"],
            "tn": tn, "fp": fp, "fn": fn, "tp": tp, "samples": len(test),
            "grid_search_seconds": record["grid_search_seconds"], "refit_seconds": record["refit_seconds"],
            "model_bytes": record["model_bytes"],
            "pipeline_ms_per_sample": record["benchmark"]["pipeline_median_ms_per_sample"],
            "end_to_end_ms_per_sample": record["benchmark"]["end_to_end_median_ms_per_sample"]})
    all_predictions = pd.concat(predictions, ignore_index=True)
    write_csv(all_predictions, OUT / "predictions_all.csv")
    metrics = pd.DataFrame(metric_rows)
    summary["highest_test_f1_descriptive_only"] = metrics.loc[metrics.f1 == metrics.f1.max(), "scenario"].tolist()
    metrics["highest_test_f1_descriptive_only"] = metrics.scenario.isin(summary["highest_test_f1_descriptive_only"])
    write_csv(metrics, OUT / "metrics_abcd.csv")
    pairs = []
    indexed = metrics.set_index("scenario")
    for first, second, factor in (("A", "C", "Underthesea"), ("B", "D", "Underthesea"), ("A", "B", "kernel"), ("C", "D", "kernel")):
        pairs.append({"pair": first + "→" + second, "factor": factor,
                      **{key + "_delta_pp": float(100 * (indexed.loc[second, key] - indexed.loc[first, key]))
                         for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "cv_f1_mean")}})
    write_csv(pd.DataFrame(pairs), OUT / "pairwise_comparisons.csv")
    summary["near_duplicates"] = sensitivity(train, test, all_predictions, OUT)
    selected_predictions = all_predictions.loc[all_predictions.scenario == selected]
    examples = []
    for outcome in ("FP", "FN"):
        errors = selected_predictions.loc[selected_predictions.outcome == outcome].copy()
        errors["abs_score"] = errors.decision_score.abs()
        for region, example in (("near_threshold", errors.nsmallest(2, "abs_score")),
                                ("far_threshold", errors.nlargest(2, "abs_score"))):
            examples.append(example.assign(region=region))
    write_csv(pd.concat(examples, ignore_index=True), OUT / "error_examples.csv")
    if hashes != {k: digest(p) for k, p in paths.items()}:
        raise ValueError("Input files changed during experiment")
    summary["completed_at"] = datetime.now(timezone(timedelta(hours=7))).isoformat()
    write_json(summary, OUT / "experiment_summary.json")
    print(f"Complete: selected={selected}; results={OUT}", flush=True)
    return summary
