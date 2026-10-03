"""Kiem tra tinh doc lap cua CV, dau vao va tai su dung mo hinh."""

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate

from src.load_data import DataConfigurationError
from src.train_svm import SCORING, build_grid_search, build_pipeline, load_training_data, make_cv_splits


class TrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        # Du lieu tong hop chi dung test; khong phai ket qua ReINTEL.
        self.data = pd.DataFrame({
            "source_index": list(range(20)),
            "processed_text": [
                f"nguon_chinh_thong rieng_{i}" if i < 10 else f"tin_don rieng_{i}"
                for i in range(20)
            ],
            "label": [0] * 10 + [1] * 10,
        })

    def test_cv_fits_vocabulary_without_validation_tokens(self) -> None:
        folds, _ = make_cv_splits(self.data)
        fitted = cross_validate(
            build_pipeline(), self.data["processed_text"], self.data["label"],
            cv=folds, return_estimator=True,
        )
        for (_, validation), estimator in zip(folds, fitted["estimator"]):
            vocabulary = estimator.named_steps["tfidf"].vocabulary_
            for index in validation:
                self.assertNotIn(f"rieng_{index}", vocabulary)

    def test_duplicate_texts_stay_in_same_fold_and_splits_reproduce(self) -> None:
        data = pd.concat([self.data, self.data.iloc[[0]].assign(source_index=20)], ignore_index=True)
        folds, cv_type = make_cv_splits(data)
        repeat, _ = make_cv_splits(data)
        self.assertEqual(cv_type, "StratifiedGroupKFold")
        validation_rows = []
        for (train, validation), (train_b, validation_b) in zip(folds, repeat):
            self.assertFalse(set(data.iloc[train]["processed_text"]) & set(data.iloc[validation]["processed_text"]))
            self.assertEqual(data.iloc[validation]["label"].nunique(), 2)
            np.testing.assert_array_equal(train, train_b)
            np.testing.assert_array_equal(validation, validation_b)
            validation_rows.extend(validation)
        self.assertEqual(sorted(validation_rows), list(range(len(data))))

    def test_grid_search_selects_by_positive_f1_and_model_reloads(self) -> None:
        folds, _ = make_cv_splits(self.data)
        for kernel in ("linear", "rbf"):
            with self.subTest(kernel=kernel):
                search = build_grid_search(kernel, folds, n_jobs=1, verbose=0)
                search.param_grid = {"svc__C": [0.1, 1.0], "svc__class_weight": [None, "balanced"]}
                search.fit(self.data["processed_text"], self.data["label"])
                self.assertEqual(search.refit, "f1")
                self.assertAlmostEqual(search.best_score_, max(search.cv_results_["mean_test_f1"]))
                self.assertEqual(len(search.cv_results_["split4_test_f1"]), 4)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "model.pkl"
                    joblib.dump(search.best_estimator_, path)
                    restored = joblib.load(path)
                    np.testing.assert_array_equal(
                        search.predict(self.data["processed_text"]),
                        restored.predict(self.data["processed_text"]),
                    )

    def test_zero_positive_predictions_produce_zero_f1(self) -> None:
        scorer = SCORING["f1"]
        model = build_pipeline().fit(self.data["processed_text"], self.data["label"])
        score = scorer(model, ["nguon_chinh_thong"] * 3, [1] * 3)
        self.assertEqual(score, 0.0)

    def test_invalid_training_input_is_rejected(self) -> None:
        cases = {
            "missing_column": self.data.drop(columns="label"),
            "null_text": self.data.assign(processed_text=[None] + self.data["processed_text"].tolist()[1:]),
            "empty_text": self.data.assign(processed_text=[" "] + self.data["processed_text"].tolist()[1:]),
            "wrong_label": self.data.assign(label=[2] + self.data["label"].tolist()[1:]),
            "duplicate_source": self.data.assign(source_index=[1] + list(range(1, 20))),
            "too_few_positive": self.data.iloc[:12],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.csv"
            for name, data in cases.items():
                with self.subTest(name=name):
                    data.to_csv(path, index=False)
                    with self.assertRaises(DataConfigurationError):
                        load_training_data(path)


if __name__ == "__main__":
    unittest.main()
