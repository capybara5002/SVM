"""Web regression tests: real inference, honest explanations, and file boundaries."""
from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
from docx import Document
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

from src.config import BASE_DIR
from src.preprocess import clean_text, segment_vietnamese
from src.train_svm import build_pipeline
from src.web_analysis import (
    AnalysisEngine, InputError, MAX_FILE_BYTES, MAX_TEXT_CHARS, cleaning_trace, extract_file,
)
from webapp import create_app


ARTICLE = "Thư viện trường thông báo sinh viên có thể mượn sách và đọc tài liệu tại trường."


def document_bytes() -> bytes:
    document = Document()
    document.add_paragraph(ARTICLE)
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Tiêu đề"
    table.cell(0, 1).text = "Nội dung bảng"
    document.add_paragraph("Đoạn cuối tài liệu.")
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def pdf_bytes(text: bool = True) -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=400, height=300)
    if text:
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                 NameObject("/Subtype"): NameObject("/Type1"),
                                 NameObject("/BaseFont"): NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
        content = DecodedStreamObject()
        content.set_data(b"BT /F1 12 Tf 20 250 Td (Thu vien thong bao cho sinh vien doc sach.) Tj ET")
        page[NameObject("/Contents")] = writer._add_object(content)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


class FileInputTests(unittest.TestCase):
    def test_unicode_text_encodings(self):
        for encoding in ("utf-8", "utf-8-sig", "utf-16"):
            text, meta = extract_file("bai-viet.txt", ARTICLE.encode(encoding))
            self.assertEqual(text, ARTICLE)
            self.assertEqual(meta["format"], "txt")

    def test_docx_preserves_paragraph_and_table_order(self):
        text, _ = extract_file("bai-viet.docx", document_bytes())
        self.assertTrue(text.startswith(ARTICLE))
        self.assertLess(text.index("Nội dung bảng"), text.index("Đoạn cuối"))

    def test_pdf_text_extraction_and_scan_error(self):
        text, meta = extract_file("article.pdf", pdf_bytes())
        self.assertIn("Thu vien", text)
        self.assertEqual(meta["pages"], 1)
        with self.assertRaisesRegex(InputError, "OCR"):
            extract_file("scan.pdf", pdf_bytes(False))

    def test_bad_files_and_limits(self):
        cases = [("bad.exe", b"text"), ("empty.txt", b""), ("bad.docx", b"not a docx"),
                 ("bad.pdf", b"not a pdf"), ("bad.txt", b"\x80\x81\x82"),
                 ("binary.txt", b"hello\x00world"), ("large.txt", b"x" * (MAX_FILE_BYTES + 1)),
                 ("long.txt", b"x" * (MAX_TEXT_CHARS + 1))]
        for name, content in cases:
            with self.subTest(name=name), self.assertRaises(InputError):
                extract_file(name, content)

    def test_cleaning_trace_matches_training_cleaner(self):
        inputs = ["  <b>TIN MỚI</b> &amp; KHÔNG sai https://example.com\n 😊 ",
                  "Vie\u0323\u0302t Nam", "www.example.com", ARTICLE]
        for text in inputs:
            trace = cleaning_trace(text)
            self.assertEqual(len(trace), 6)
            self.assertEqual(trace[-1]["text"], clean_text(text))
            self.assertTrue(all(isinstance(step["changed"], bool) for step in trace))


class WebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        (cls.root / "artifacts/chapter3_4").mkdir(parents=True)
        (cls.root / "results/chapter3_4").mkdir(parents=True)
        texts = [ARTICLE, "Trường thông báo lịch học và thư viện mở cửa cho sinh viên.",
                 "Sinh viên có thể đọc sách tại thư viện trường theo lịch thông báo.",
                 "Thông báo mới về chương trình đào tạo và tài liệu học tập.",
                 "Tin đồn bí mật gây sốc chia sẻ ngay cho mọi người.",
                 "Sự thật bí mật gây sốc tin đồn lan truyền chưa xác minh.",
                 "Chia sẻ ngay tin đồn gây sốc này cho tất cả mọi người.",
                 "Bí mật động trời chưa xác minh đang lan truyền trên mạng."]
        labels = [0, 0, 0, 0, 1, 1, 1, 1]
        records = []
        for identifier, segmented, kernel in [("A", False, "linear"), ("B", False, "rbf"),
                                               ("C", True, "linear"), ("D", True, "rbf")]:
            transform = segment_vietnamese if segmented else clean_text
            model = build_pipeline(kernel)
            model.fit([transform(text) for text in texts], labels)
            path = cls.root / f"artifacts/chapter3_4/svm_{identifier}.pkl"
            joblib.dump(model, path)
            records.append({"scenario": identifier, "underthesea": segmented, "kernel": kernel,
                            "cv_f1_mean": 0.5, "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        (cls.root / "results/chapter3_4/experiment_summary.json").write_text(
            json.dumps({"models": records, "selected_by_cv": "D"}), encoding="utf-8")
        cls.engine = AnalysisEngine(cls.root)
        cls.app = create_app(cls.engine)
        cls.app.testing = True

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.client = self.app.test_client()

    def streamed(self, data):
        response = self.client.post("/api/analyze", data=data)
        self.assertEqual(response.status_code, 200)
        return [json.loads(line) for line in response.data.decode("utf-8").splitlines()]

    def test_page_assets_and_model_catalog(self):
        for url, mime in [("/", "text/html"), ("/static/app.js", "javascript"),
                          ("/static/style.css", "text/css"), ("/static/favicon.svg", "image/svg+xml")]:
            with self.subTest(url=url):
                response = self.client.get(url)
                self.assertEqual(response.status_code, 200)
                self.assertIn(mime, response.content_type)
                self.assertIn("no-store", response.headers["Cache-Control"])
                response.close()
        catalog = self.client.get("/api/models").json
        self.assertEqual(catalog["default"], "D")
        self.assertEqual([model["id"] for model in catalog["models"]], list("ABCD"))
        self.assertTrue(all("path" not in model for model in catalog["models"]))

    def test_stream_matches_pipeline_all_models_without_fitting(self):
        for identifier in "ABCD":
            with self.subTest(model=identifier):
                model = self.engine.load(identifier)
                transform = segment_vietnamese if identifier in "CD" else clean_text
                expected_input = transform(ARTICLE)
                expected_label = int(model.predict([expected_input])[0])
                expected_score = float(model.decision_function([expected_input])[0])
                before = dict(model.named_steps["tfidf"].vocabulary_)
                with patch.object(TfidfVectorizer, "fit", side_effect=AssertionError("Must not fit")), \
                     patch.object(TfidfVectorizer, "fit_transform", side_effect=AssertionError("Must not fit")), \
                     patch.object(SVC, "fit", side_effect=AssertionError("Must not fit")):
                    events = self.streamed({"text": ARTICLE, "model": identifier})
                self.assertEqual([(event["stage"], event["status"]) for event in events[:-1]],
                    [(stage, status) for stage in ["input", "clean", "segment", "tfidf", "svm"]
                     for status in ["running", "complete"]])
                final = events[-1]
                self.assertEqual(final["type"], "complete")
                result = final["result"]
                self.assertEqual(result["label"], expected_label)
                self.assertAlmostEqual(result["decision_score"], expected_score, places=10)
                self.assertAlmostEqual(result["decision_score"], result["bias"] + result["explanation"]["contribution_sum"], places=8)
                segmented = final["stages"][2]["data"]
                self.assertEqual(segmented["text"], expected_input)
                self.assertEqual(segmented["enabled"], identifier in "CD")
                vectorized = final["stages"][3]["data"]
                self.assertEqual(vectorized["active_features"], len(vectorized["features"]))
                self.assertAlmostEqual(sum(f["weight"] ** 2 for f in vectorized["features"]), 1.0)
                self.assertEqual(before, model.named_steps["tfidf"].vocabulary_)

    def test_txt_docx_and_pdf_uploads(self):
        for name, content in [("input.txt", ARTICLE.encode("utf-8")), ("input.docx", document_bytes()),
                              ("input.pdf", pdf_bytes())]:
            with self.subTest(name=name):
                events = self.streamed({"mode": "file", "model": "D", "file": (io.BytesIO(content), name)})
                self.assertEqual(events[-1]["type"], "complete")
                self.assertEqual(events[-1]["stages"][0]["data"]["source"]["filename"], name)

    def test_empty_unknown_model_missing_upload_and_limits(self):
        cases = [{"text": "  "}, {"text": ARTICLE, "model": "../../bad"},
                 {"mode": "file"}, {"mode": "other"}, {"text": "x" * (MAX_TEXT_CHARS + 1)},
                 {"mode": "file", "file": (io.BytesIO(b"bad"), "article.exe")}]
        for data in cases:
            with self.subTest(keys=list(data)):
                response = self.client.post("/api/analyze", data=data)
                self.assertEqual(response.status_code, 400)
                self.assertIn("error", response.json)
        response = self.client.post("/api/analyze", data=b"x" * (MAX_FILE_BYTES + 128 * 1024 + 1),
                                    content_type="application/octet-stream")
        self.assertEqual(response.status_code, 413)

    def test_unusable_content_never_produces_a_label(self):
        for data in [{"text": "<b></b> https://example.com"}, {"text": "zzzxxyyqqqwwvvvv"},
                     {"mode": "file", "file": (io.BytesIO(pdf_bytes(False)), "scan.pdf")}]:
            events = self.streamed(data)
            self.assertEqual(events[-1]["type"], "error")
            self.assertFalse(any(event["type"] == "complete" for event in events))

    def test_cross_origin_submission_rejected(self):
        response = self.client.post("/api/analyze", data={"text": ARTICLE}, headers={"Origin": "https://example.com"})
        self.assertEqual(response.status_code, 403)

    def test_missing_models_and_changed_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            empty = AnalysisEngine(Path(directory))
            client = create_app(empty).test_client()
            self.assertEqual(client.get("/api/models").json["models"], [])
            self.assertEqual(client.post("/api/analyze", data={"text": ARTICLE}).status_code, 503)
        fresh = AnalysisEngine(self.root)
        fresh.specs["D"]["sha256"] = "different"
        with self.assertRaisesRegex(RuntimeError, "metadata"):
            fresh.load("D")


class RepositoryModelTests(unittest.TestCase):
    @unittest.skipUnless((BASE_DIR / "artifacts/chapter3_4/svm_D.pkl").exists(), "Saved research models are optional")
    def test_saved_model_and_explanation_agree(self):
        engine = AnalysisEngine()
        for identifier in engine.specs:
            with self.subTest(model=identifier):
                final = list(engine.events(identifier, ARTICLE))[-1]
                model = engine.load(identifier)
                transform = segment_vietnamese if engine.specs[identifier]["segmented"] else clean_text
                self.assertEqual(final["result"]["label"], int(model.predict([transform(ARTICLE)])[0]))
                result = final["result"]
                self.assertAlmostEqual(result["decision_score"], result["bias"] + result["explanation"]["contribution_sum"], places=8)


if __name__ == "__main__":
    unittest.main()
