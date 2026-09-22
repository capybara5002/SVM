import unittest

import numpy as np

from src.preprocess import clean_text


class CleanTextTests(unittest.TestCase):
    def test_remove_url(self) -> None:
        result = clean_text("Tin mới tại https://example.com/bai-viet hôm nay")
        self.assertNotIn("https://", result)
        self.assertEqual(result, "tin mới tại hôm nay")

    def test_normalize_whitespace(self) -> None:
        self.assertEqual(clean_text("  Xin\n\t chào   Việt Nam  "), "xin chào việt nam")

    def test_empty_and_null_input(self) -> None:
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text(np.nan), "")
        self.assertEqual(clean_text(""), "")

    def test_keep_vietnamese_unicode_and_negation(self) -> None:
        result = clean_text("KHÔNG xóa dấu tiếng Việt: chẳng, chưa!")
        self.assertEqual(result, "không xóa dấu tiếng việt: chẳng, chưa!")

    def test_remove_html_but_keep_punctuation(self) -> None:
        self.assertEqual(clean_text("<b>Chào bạn!</b> 😊"), "chào bạn! 😊")


if __name__ == "__main__":
    unittest.main()

