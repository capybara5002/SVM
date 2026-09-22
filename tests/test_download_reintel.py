import unittest

from download_reintel import verify_label_mapping_from_card


class DatasetCardTests(unittest.TestCase):
    def test_verify_official_label_mapping(self) -> None:
        card = "label:\n* 1: unreliable\n* 0: reliable\n"
        self.assertEqual(verify_label_mapping_from_card(card), {0: "reliable", 1: "unreliable"})

    def test_reject_unverified_label_mapping(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Không xác minh được"):
            verify_label_mapping_from_card("label is numeric")


if __name__ == "__main__":
    unittest.main()
