import unittest
from unittest.mock import patch

import pandas as pd

from src.load_data import DataConfigurationError, identify_columns


class IdentifyColumnsTests(unittest.TestCase):
    @patch("src.load_data.config.LABEL_COLUMN", None)
    @patch("src.load_data.config.TEXT_COLUMN", None)
    def test_identify_unambiguous_common_columns(self) -> None:
        dataframe = pd.DataFrame({"content": ["Tin mẫu"], "label": ["reliable"]})
        self.assertEqual(identify_columns(dataframe), ("content", "label"))

    @patch("src.load_data.config.LABEL_COLUMN", None)
    @patch("src.load_data.config.TEXT_COLUMN", None)
    def test_do_not_guess_when_text_columns_are_ambiguous(self) -> None:
        dataframe = pd.DataFrame(
            {"text": ["Tin mẫu"], "content": ["Nội dung"], "label": ["reliable"]}
        )
        with self.assertRaisesRegex(DataConfigurationError, "TEXT_COLUMN"):
            identify_columns(dataframe)

    @patch("src.load_data.config.LABEL_COLUMN", "nhan_that")
    @patch("src.load_data.config.TEXT_COLUMN", "noi_dung_that")
    def test_honor_manually_configured_columns(self) -> None:
        dataframe = pd.DataFrame({"noi_dung_that": ["Tin mẫu"], "nhan_that": [1]})
        self.assertEqual(identify_columns(dataframe), ("noi_dung_that", "nhan_that"))


if __name__ == "__main__":
    unittest.main()
