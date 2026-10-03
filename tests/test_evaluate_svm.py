"""Kiem tra danh gia holdout bang du lieu tong hop, khong dung so lieu ReINTEL."""

import hashlib
import io
import json
import shutil
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

import joblib
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.evaluate_svm import compute_metrics, evaluate_saved_models, load_processed_data, prepare_test_set
from src.load_data import DataConfigurationError
from src.train_svm import build_pipeline


class EvaluationMetricTests(unittest.TestCase):
    def test_confusion_matrix_and_metrics_use_unreliable_as_positive(self) -> None:
        # TN=3, FP=1, FN=2, TP=2; the errors are asymmetric on purpose.
        result = compute_metrics(
            [0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 1, 0, 0, 1, 1],
        )
        self.assertEqual(result["confusion_matrix"], [[3, 1], [2, 2]])
        expected = {
            "accuracy": 5 / 8,
            "precision": 2 / 3,
            "recall": 1 / 2,
            "f1": 4 / 7,
            "f1_macro": ((2 / 3) + (4 / 7)) / 2,
            "f1_weighted": ((2 / 3) + (4 / 7)) / 2,
            "balanced_accuracy": ((3 / 4) + (2 / 4)) / 2,
        }
        for name, value in expected.items():
            with self.subTest(metric=name):
                self.assertAlmostEqual(result["metrics"][name], value)
        report = result["classification_report"]
        self.assertEqual(report["reliable"]["support"], 4)
        self.assertEqual(report["unreliable"]["support"], 4)
        self.assertAlmostEqual(report["unreliable"]["precision"], 2 / 3)
        self.assertAlmostEqual(report["unreliable"]["recall"], 1 / 2)

    def test_no_positive_predictions_have_zero_positive_metrics(self) -> None:
        result = compute_metrics([0, 0, 0, 1], [0, 0, 0, 0])
        self.assertEqual(result["confusion_matrix"], [[3, 0], [1, 0]])
        self.assertEqual(result["metrics"]["accuracy"], 0.75)
        for name in ("precision", "recall", "f1"):
            self.assertEqual(result["metrics"][name], 0.0)
        # Majority-class accuracy must not hide failure on the positive class.
        self.assertEqual(result["metrics"]["balanced_accuracy"], 0.5)

    def test_invalid_prediction_vectors_are_rejected(self) -> None:
        cases = [([], []), ([0, 1], [0]), ([[0, 1]], [[0, 1]]), ([0, 2], [0, 1]), ([0, 1], [0, 3])]
        for truth, prediction in cases:
            with self.subTest(truth=truth, prediction=prediction):
                with self.assertRaises(ValueError):
                    compute_metrics(truth, prediction)


class EvaluationInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = pd.DataFrame(
            {
                "source_index": [11, 12, 13, 14],
                "processed_text": ["tin_chinh_thong a", "tin_don b", "tin_chinh_thong c", "tin_don d"],
                "label": [0, 1, 0, 1],
            }
        )

    def test_valid_processed_csv_retains_source_identity_and_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.csv"
            self.data.to_csv(path, index=False, encoding="utf-8-sig")
            loaded = load_processed_data(path, "test")
        assert_frame_equal(loaded, self.data)

    def test_invalid_processed_inputs_are_rejected(self) -> None:
        cases = {
            "missing_column": self.data.drop(columns="source_index"),
            "null_text": self.data.assign(processed_text=[None, "b", "c", "d"]),
            "null_label": self.data.assign(label=[None, 1, 0, 1]),
            "null_source": self.data.assign(source_index=[None, 12, 13, 14]),
            "empty_text": self.data.assign(processed_text=[" ", "b", "c", "d"]),
            "numeric_text": self.data.assign(processed_text=[100, 101, 102, 103]),
            "unknown_label": self.data.assign(label=[2, 1, 0, 1]),
            "one_label": self.data.assign(label=[0, 0, 0, 0]),
            "duplicate_source": self.data.assign(source_index=[11, 11, 13, 14]),
            "empty_data": self.data.iloc[:0],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.csv"
            for name, data in cases.items():
                with self.subTest(case=name):
                    data.to_csv(path, index=False, encoding="utf-8-sig")
                    with self.assertRaises(DataConfigurationError):
                        load_processed_data(path, "test")


class HoldoutAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        # The shared training text occurs at two nonascending source IDs.
        self.train = pd.DataFrame(
            {
                "source_index": [30, 2, 10],
                "processed_text": ["shared", "shared", "training_positive"],
                "label": [0, 0, 1],
            }
        )
        self.test = pd.DataFrame(
            {
                "source_index": [100, 101, 102, 103, 104, 105],
                "processed_text": ["shared", "heldout_negative", "heldout_positive", "heldout_negative", "shared", "other_positive"],
                "label": [0, 0, 1, 0, 0, 1],
            }
        )

    def test_overlap_and_duplicate_exclusions_preserve_first_unseen_example(self) -> None:
        clean, excluded, audit = prepare_test_set(self.train, self.test)
        self.assertEqual(clean["source_index"].tolist(), [101, 102, 105])
        self.assertFalse(set(clean["processed_text"]) & set(self.train["processed_text"]))
        self.assertFalse(clean["processed_text"].duplicated().any())
        self.assertEqual(set(excluded["source_index"]), {100, 103, 104})
        exclusions = excluded.set_index("source_index")
        for source in (100, 104):
            self.assertEqual(exclusions.loc[source, "exclusion_reason"], "train_text_overlap")
            self.assertEqual(exclusions.loc[source, "matching_train_source_indices"], "2;30")
        self.assertEqual(exclusions.loc[103, "exclusion_reason"], "duplicate_test_text")
        self.assertEqual(audit["original_test_rows"], 6)
        self.assertEqual(audit["evaluated_test_rows"], 3)
        self.assertEqual(audit["excluded_overlap_rows"], 2)
        self.assertEqual(audit["excluded_duplicate_rows"], 1)
        self.assertEqual(audit["evaluated_test_labels"], {"0": 1, "1": 2})
        self.assertEqual(len(clean) + len(excluded), len(self.test))

    def test_audit_is_deterministic_and_does_not_modify_inputs(self) -> None:
        original_train = self.train.copy(deep=True)
        original_test = self.test.copy(deep=True)
        clean, excluded, audit = prepare_test_set(self.train, self.test)
        repeated_clean, repeated_excluded, repeated_audit = prepare_test_set(self.train, self.test)
        assert_frame_equal(self.train, original_train)
        assert_frame_equal(self.test, original_test)
        assert_frame_equal(clean, repeated_clean)
        assert_frame_equal(excluded, repeated_excluded)
        self.assertEqual(audit, repeated_audit)

    def test_source_identity_overlap_is_a_configuration_error(self) -> None:
        changed = self.test.copy(deep=True)
        changed.loc[0, "source_index"] = 30
        changed.loc[0, "processed_text"] = "different_text_same_source"
        with self.assertRaises(DataConfigurationError):
            prepare_test_set(self.train, changed)

    def test_conflicting_test_labels_are_rejected_before_deduplication(self) -> None:
        changed = self.test.copy(deep=True)
        changed.loc[3, "label"] = 1
        with self.assertRaises(DataConfigurationError):
            prepare_test_set(self.train, changed)

    def test_filtering_cannot_leave_only_one_test_class(self) -> None:
        changed = self.test.copy(deep=True)
        changed.loc[changed["label"].eq(1), "processed_text"] = "training_positive"
        with self.assertRaises(DataConfigurationError):
            prepare_test_set(self.train, changed)


class FrozenModelEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        # This deliberately makes the CV-selected RBF worse than linear on test.
        # It checks retention of the preselected model, not model-selection logic.
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.artifacts = self.root / "artifacts"
        self.artifacts.mkdir()
        self.train_path = self.root / "train.csv"
        self.test_path = self.root / "test.csv"
        self.summary_path = self.root / "training_summary.json"
        self.docs = self.root / "docs"
        self.report = self.root / "results" / "week6.md"
        train = pd.DataFrame({
            "source_index": list(range(8)),
            "processed_text": [f"trusted official reporting source entry{i}" for i in range(6)]
                + [f"fake rumor unreliable secret entry{i}" for i in range(2)],
            "label": [0] * 6 + [1] * 2,
        })
        test = pd.DataFrame({
            "source_index": list(range(100, 106)),
            "processed_text": [train.loc[0, "processed_text"], "trusted official news heldout",
                "fake rumor secret heldout", "trusted official news heldout",
                "official trusted source extra", "unreliable fake secret extra"],
            "label": [0, 0, 1, 0, 0, 1],
        })
        train.to_csv(self.train_path, index=False, encoding="utf-8-sig")
        test.to_csv(self.test_path, index=False, encoding="utf-8-sig")
        settings = {
            "svm_linear_baseline": ("linear", 10.0, "scale", None, "svm_linear_baseline.pkl"),
            "svm_rbf_baseline": ("rbf", 1.0, "scale", None, "svm_rbf_baseline.pkl"),
            "svm_linear_tuned": ("linear", 20.0, "scale", None, "svm_linear_best.pkl"),
            "svm_rbf_tuned": ("rbf", 0.001, 100.0, None, "svm_rbf_best.pkl"),
        }
        metadata = {
            "train_sha256": hashlib.sha256(self.train_path.read_bytes()).hexdigest(),
            "train_rows": len(train),
            "baselines": {}, "searches": {},
            "best_model": {"kernel": "rbf", "mean_cv_f1": 0.9},
        }
        for model_id, (kernel, penalty, gamma, class_weight, filename) in settings.items():
            params = {"kernel": kernel, "C": penalty, "gamma": gamma, "class_weight": class_weight}
            model = build_pipeline(kernel).set_params(
                svc__C=penalty, svc__gamma=gamma, svc__class_weight=class_weight,
            )
            model.fit(train["processed_text"], train["label"])
            joblib.dump(model, self.artifacts / filename)
            record = {"metrics": {"f1": {"mean": 0.9 if model_id == "svm_rbf_tuned" else 0.5}}}
            if model_id.endswith("baseline"):
                metadata["baselines"][model_id] = {**record, "params": params}
            else:
                metadata["searches"][kernel] = {
                    **record, "best_params": {"svc__C": penalty, "svc__gamma": gamma,
                        "svc__class_weight": class_weight},
                }
        shutil.copyfile(self.artifacts / "svm_rbf_best.pkl", self.artifacts / "svm_best_model.pkl")
        self.summary_path.write_text(json.dumps(metadata), encoding="utf-8")
        self.kwargs = {
            "train_path": self.train_path, "test_path": self.test_path,
            "training_summary_path": self.summary_path, "artifacts_dir": self.artifacts,
            "docs_dir": self.docs, "report_path": self.report,
        }

    def test_evaluation_predicts_without_refitting_or_reselecting_and_exports_rows(self) -> None:
        frozen_files = [self.train_path, self.test_path, self.summary_path, *self.artifacts.glob("*.pkl")]
        original_bytes = {path: path.read_bytes() for path in frozen_files}
        visuals = types.ModuleType("src.evaluation_visuals")
        visuals.render_evaluation_docs = Mock()
        with ExitStack() as stack:
            stack.enter_context(patch("sys.stdout", io.StringIO()))
            stack.enter_context(patch.dict(sys.modules, {"src.evaluation_visuals": visuals}))
            for model_class, method in (
                (Pipeline, "fit"), (TfidfVectorizer, "fit"),
                (TfidfVectorizer, "fit_transform"), (SVC, "fit"),
            ):
                stack.enter_context(patch.object(
                    model_class, method, side_effect=AssertionError("Test evaluation must not refit"),
                ))
            summary = evaluate_saved_models(**self.kwargs)
        visuals.render_evaluation_docs.assert_called_once_with(summary, self.docs)
        self.assertEqual(summary["selected_model_id"], "svm_rbf_tuned")
        self.assertEqual(summary["model_selection"]["source"], "week4_5_cv")
        self.assertFalse(summary["model_selection"]["test_used_for_selection"])
        self.assertFalse(summary["model_selection"]["refit_during_evaluation"])
        scores = {model["id"]: model["metrics"]["f1"] for model in summary["models"]}
        self.assertGreater(scores["svm_linear_baseline"], scores["svm_rbf_tuned"])
        self.assertEqual(summary["data_audit"]["evaluated_test_rows"], 4)
        for path, content in original_bytes.items():
            self.assertEqual(path.read_bytes(), content, str(path))
        predictions = pd.read_csv(self.docs / "predictions" / "all_predictions.csv")
        self.assertEqual(len(predictions), 16)
        self.assertEqual(set(predictions["source_index"]), {101, 102, 104, 105})
        self.assertEqual(set(predictions["outcome"]), {"TN", "FN", "TP"})
        metrics = pd.read_csv(self.docs / "tables" / "model_metrics.csv")
        self.assertEqual(metrics.loc[metrics["selected_by_cv"], "model_id"].tolist(), ["svm_rbf_tuned"])
        for model in summary["models"]:
            rows = predictions.loc[predictions["model_id"].eq(model["id"])]
            self.assertEqual(compute_metrics(rows["y_true"], rows["y_pred"])["confusion_matrix"], model["confusion_matrix"])
            self.assertEqual(model["evaluated_samples"], 4)
        self.assertTrue(self.report.is_file())
        saved = json.loads((self.docs / "evaluation_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["selected_model_id"], summary["selected_model_id"])

    def test_changed_training_file_is_refused_before_model_loading(self) -> None:
        with self.train_path.open("ab") as stream:
            stream.write(b"\n")
        with patch("src.evaluate_svm.joblib.load") as load_model:
            with self.assertRaises(DataConfigurationError):
                evaluate_saved_models(**self.kwargs)
            load_model.assert_not_called()
        self.assertFalse(self.docs.exists())

    def test_selected_alias_must_match_the_saved_cv_winner(self) -> None:
        shutil.copyfile(self.artifacts / "svm_linear_best.pkl", self.artifacts / "svm_best_model.pkl")
        with patch("src.evaluate_svm.joblib.load") as load_model:
            with self.assertRaises(DataConfigurationError):
                evaluate_saved_models(**self.kwargs)
            load_model.assert_not_called()
        self.assertFalse(self.docs.exists())


if __name__ == "__main__":
    unittest.main()
