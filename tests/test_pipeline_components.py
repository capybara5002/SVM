import unittest

import pandas as pd

from src.split_data import split_dataframe
from src.vectorize import fit_tfidf_on_train


class PipelineComponentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = pd.DataFrame(
            {
                "text_segmented": [f"tin_{index} chung" for index in range(20)],
                "label": ["reliable"] * 10 + ["unreliable"] * 10,
            }
        )

    def test_stratified_split_is_80_20_and_reproducible(self) -> None:
        train_a, test_a = split_dataframe(self.data, "label")
        train_b, test_b = split_dataframe(self.data, "label")
        self.assertEqual((len(train_a), len(test_a)), (16, 4))
        self.assertEqual(train_a.index.tolist(), train_b.index.tolist())
        self.assertEqual(test_a["label"].value_counts().to_dict(), {"reliable": 2, "unreliable": 2})

    def test_tfidf_fits_only_training_vocabulary(self) -> None:
        train = ["tin chung đáng_tin", "nguồn chung chính_thống"]
        test = ["tu_chi_xuat_hien_o_test chung"]
        vectorizer, train_matrix, test_matrix = fit_tfidf_on_train(train, test)
        self.assertNotIn("tu_chi_xuat_hien_o_test", vectorizer.vocabulary_)
        self.assertEqual(train_matrix.shape[1], test_matrix.shape[1])
        self.assertGreater(test_matrix.nnz, 0)  # "chung" da co trong tap train.


if __name__ == "__main__":
    unittest.main()

