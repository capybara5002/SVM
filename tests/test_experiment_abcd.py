"""Protocol tests: source identity, shared clean-text groups, fold-local vocabulary."""
import unittest
import numpy as np
import pandas as pd
from sklearn.base import clone
from src.experiment_abcd import audit_inputs
from src.train_svm import build_pipeline
from src.vectorize import create_tfidf_vectorizer


class ExperimentProtocolTests(unittest.TestCase):
    def setUp(self):
        self.raw = pd.DataFrame({"text": [f"nội dung {i}" for i in range(14)], "label": [0, 1] * 7})
        self.train = pd.DataFrame({"source_index": range(10), "processed_text": [f"nội_dung {i}" for i in range(10)], "label": [0, 1] * 5})
        self.test = pd.DataFrame({"source_index": range(10,14), "processed_text": [f"nội_dung {i}" for i in range(10,14)], "label": [0,1]*2})
        self.folds = self.train[["source_index", "label"]].assign(validation_fold=np.repeat(range(1,6),2))

    def audit(self):
        return audit_inputs(self.train, self.test, self.raw, self.folds, (10,4))

    def test_keeps_shared_id_order_and_all_five_partitions(self):
        train, test, splits, audit = self.audit()
        self.assertEqual(train.source_index.tolist(), list(range(10)))
        self.assertEqual(test.source_index.tolist(), list(range(10,14)))
        self.assertEqual(sorted(np.concatenate([v for _,v in splits]).tolist()), list(range(10)))
        self.assertEqual(audit["clean_text_overlap"],0)

    def test_rejects_clean_overlap_even_when_segmented_strings_differ(self):
        self.raw.loc[10,"text"] = self.raw.loc[0,"text"]
        with self.assertRaisesRegex(ValueError,"cleaned holdout"):
            self.audit()

    def test_rejects_clean_overlap_across_fold_boundaries(self):
        self.raw.loc[2,"text"] = self.raw.loc[0,"text"]
        with self.assertRaisesRegex(ValueError,"Fold 1: text_clean leakage"):
            self.audit()

    def test_rejects_label_and_id_changes(self):
        self.folds.loc[0,"label"] = 1
        with self.assertRaisesRegex(ValueError,"Fold labels"):
            self.audit()
        self.folds.loc[0,"source_index"] = 1
        with self.assertRaisesRegex(ValueError,"one-to-one"):
            self.audit()

    def test_no_underthesea_is_unicode_regex_not_whitespace_split(self):
        tokens = create_tfidf_vectorizer().build_analyzer()("học sinh, a 12 học_sinh!")
        self.assertEqual(tokens,["học","sinh","a","12","học_sinh"])

    def test_validation_only_token_does_not_enter_fold_vocabulary(self):
        pipeline = clone(build_pipeline()).fit(["tin chính thức", "tin đồn sai"],[0,1])
        pipeline.predict(["validation_only_token tin"])
        self.assertNotIn("validation_only_token",pipeline.named_steps["tfidf"].vocabulary_)


if __name__ == "__main__":
    unittest.main()
