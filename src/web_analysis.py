"""Inference and inspectable stages for the local demo; never fit on user input."""
from __future__ import annotations

import hashlib
import html
import io
import json
import threading
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

from .config import BASE_DIR
from .preprocess import (
    HTML_TAG_PATTERN, URL_PATTERN, WHITESPACE_PATTERN, clean_text, segment_vietnamese,
)

MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_TEXT_CHARS = 100_000
EXTENSIONS = {".txt", ".md", ".docx", ".pdf"}
LABELS = {0: "Thông tin đáng tin cậy", 1: "Thông tin không đáng tin cậy"}
DISCLAIMER = (
    "Đây là dự đoán thống kê từ nội dung văn bản, không phải kết luận kiểm chứng sự thật. "
    "Mô hình học từ bài đăng mạng xã hội ReINTEL; kết quả có thể kém chính xác với "
    "bài báo dài, chủ đề mới hoặc văn bản khác miền dữ liệu. Điểm SVM không phải xác suất."
)


class InputError(ValueError):
    """A readable, expected error in submitted content."""


def validate_text(text: str) -> str:
    if not text or not text.strip():
        raise InputError("Hãy dán nội dung hoặc chọn một tệp có văn bản để phân tích.")
    if len(text) > MAX_TEXT_CHARS:
        raise InputError("Nội dung quá dài. Mỗi lần phân tích tối đa 100.000 ký tự.")
    if "\x00" in text:
        raise InputError("Nội dung chứa ký tự nhị phân. Hãy dùng tệp văn bản UTF-8.")
    return text


def extract_file(name: str, content: bytes) -> tuple[str, dict]:
    """Extract in memory; do not persist uploads or run document macros."""
    suffix = Path(name).suffix.lower()
    if suffix not in EXTENSIONS:
        raise InputError("Chỉ hỗ trợ tệp .txt, .md, .docx và .pdf có lớp văn bản.")
    if len(content) > MAX_FILE_BYTES:
        raise InputError("Tệp vượt quá giới hạn 5 MB.")
    if not content:
        raise InputError("Tệp đang trống. Hãy chọn tệp khác.")
    metadata = {"filename": name, "bytes": len(content), "format": suffix[1:]}
    try:
        if suffix in {".txt", ".md"}:
            encoding = "utf-16" if content.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
            text = content.decode(encoding)
            metadata["encoding"] = encoding
        elif suffix == ".docx":
            from docx import Document
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                if sum(item.file_size for item in archive.infolist()) > 25 * 1024 * 1024:
                    raise InputError("Tệp DOCX giải nén vượt quá giới hạn 25 MB.")
            document = Document(io.BytesIO(content))
            # Preserve paragraph/table order in the document body.
            from docx.table import Table
            from docx.text.paragraph import Paragraph
            blocks = []
            for block in document.iter_inner_content():
                if isinstance(block, Paragraph):
                    blocks.append(block.text)
                elif isinstance(block, Table):
                    blocks.extend("\t".join(cell.text for cell in row.cells) for row in block.rows)
            text = "\n".join(blocks)
        else:
            from pypdf import PdfReader
            if not content.lstrip().startswith(b"%PDF-"):
                raise InputError("Tệp không có định dạng PDF hợp lệ. Hãy chọn đúng tệp PDF.")
            reader = PdfReader(io.BytesIO(content))
            if reader.is_encrypted:
                raise InputError("PDF có mật khẩu. Hãy mở khóa tệp trước khi tải lên.")
            if len(reader.pages) > 80:
                raise InputError("Mỗi lần phân tích tối đa 80 trang PDF.")
            metadata["pages"] = len(reader.pages)
            pages, size = [], 0
            for page in reader.pages:
                value = page.extract_text() or ""
                size += len(value) + 1
                if size > MAX_TEXT_CHARS:
                    raise InputError("PDF chứa hơn 100.000 ký tự. Hãy tách thành tệp nhỏ hơn.")
                pages.append(value)
            text = "\n".join(pages)
            if not text.strip():
                raise InputError("PDF không có văn bản đọc được. PDF ảnh/scan cần được OCR trước.")
    except InputError:
        raise
    except UnicodeError as error:
        raise InputError("Không đọc được mã hóa tệp. Hãy lưu TXT/MD bằng UTF-8 hoặc UTF-16 có BOM.") from error
    except Exception as error:
        raise InputError("Không đọc được tệp. Hãy kiểm tra tệp có đúng định dạng và không bị hỏng.") from error
    return validate_text(text), metadata


def cleaning_trace(text: str) -> list[dict]:
    """Expose the exact operations used by clean_text, including intermediate text."""
    original = text
    operations = [
        ("Chuẩn hóa Unicode NFC", lambda s: unicodedata.normalize("NFC", s)),
        ("Giải mã HTML entity", html.unescape),
        ("Loại bỏ thẻ HTML", lambda s: HTML_TAG_PATTERN.sub(" ", s)),
        ("Loại bỏ đường dẫn URL", lambda s: URL_PATTERN.sub(" ", s)),
        ("Chuyển về chữ thường", str.lower),
        ("Gom khoảng trắng", lambda s: WHITESPACE_PATTERN.sub(" ", s).strip()),
    ]
    trace = []
    for name, transform in operations:
        previous = text
        text = transform(text)
        trace.append({"name": name, "changed": previous != text, "text": text})
    if text != clean_text(original):
        raise RuntimeError("Cleaning trace differs from the training preprocessor")
    return trace


class AnalysisEngine:
    def __init__(self, root: Path = BASE_DIR):
        self.root = Path(root)
        self._models = {}
        self._lock = threading.Lock()
        summary_path = self.root / "results/chapter3_4/experiment_summary.json"
        self.specs = {}
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            for record in summary["models"]:
                identifier = record["scenario"]
                path = self.root / "artifacts/chapter3_4" / f"svm_{identifier}.pkl"
                if identifier in {"A", "B", "C", "D"} and path.exists():
                    self.specs[identifier] = {
                        "id": identifier, "name": f"{identifier} · SVM {record['kernel'].upper()}",
                        "segmented": record["underthesea"], "kernel": record["kernel"],
                        "cv_f1": record["cv_f1_mean"], "path": path,
                        "sha256": record.get("model_sha256"),
                    }
            self.default = summary.get("selected_by_cv", "D")
        else:
            self.default = "best"
        legacy = self.root / "artifacts/svm_best_model.pkl"
        if not self.specs and legacy.exists():
            self.specs["best"] = {
                "id": "best", "name": "SVM tốt nhất · Tuần 5", "segmented": True,
                "kernel": None, "cv_f1": None, "path": legacy, "sha256": None,
            }
        if self.default not in self.specs:
            self.default = next(iter(self.specs), None)

    def catalog(self) -> dict:
        return {"default": self.default, "models": [
            {key: value for key, value in spec.items() if key not in {"path", "sha256"}}
            for spec in self.specs.values()
        ], "limits": {"file_bytes": MAX_FILE_BYTES, "text_chars": MAX_TEXT_CHARS},
            "disclaimer": DISCLAIMER}

    def load(self, identifier: str):
        if identifier not in self.specs:
            raise InputError("Mô hình không khả dụng. Hãy tải lại trang và chọn mô hình có sẵn.")
        with self._lock:
            if identifier not in self._models:
                spec = self.specs[identifier]
                if spec["sha256"] and hashlib.sha256(spec["path"].read_bytes()).hexdigest() != spec["sha256"]:
                    raise RuntimeError("Model artifact does not match experiment metadata")
                model = joblib.load(spec["path"])
                if list(model.classes_) != [0, 1] or set(model.named_steps) != {"tfidf", "svc"}:
                    raise RuntimeError("Expected a fitted TF-IDF/SVC pipeline with classes [0, 1]")
                self._models[identifier] = model
            return self._models[identifier]

    def events(self, identifier: str, text: str = "", filename: str | None = None, content: bytes = b""):
        """NDJSON-friendly real stage events. No artificial progress or training."""
        overall_start = perf_counter()
        stages = []
        spec = self.specs[identifier]
        warnings = []

        def finished(stage, started, data):
            result = {"type": "stage", "stage": stage, "status": "complete",
                      "duration_ms": round((perf_counter() - started) * 1000, 3), "data": data}
            stages.append(result)
            return result

        yield {"type": "stage", "stage": "input", "status": "running"}
        start = perf_counter()
        source = {"format": "text", "filename": None}
        if filename is not None:
            text, source = extract_file(filename, content)
        else:
            text = validate_text(text)
        yield finished("input", start, {"text": text, "characters": len(text),
                                         "whitespace_words": len(text.split()), "source": source})

        yield {"type": "stage", "stage": "clean", "status": "running"}
        start = perf_counter()
        trace = cleaning_trace(text)
        cleaned = trace[-1]["text"]
        if not cleaned:
            raise InputError("Nội dung rỗng sau làm sạch (chỉ có URL hoặc thẻ HTML). Hãy thêm nội dung bài viết.")
        yield finished("clean", start, {"text": cleaned, "characters": len(cleaned), "operations": trace})

        yield {"type": "stage", "stage": "segment", "status": "running"}
        start = perf_counter()
        processed = segment_vietnamese(text) if spec["segmented"] else cleaned
        yield finished("segment", start, {"text": processed, "enabled": spec["segmented"],
            "tokens": processed.split(), "token_count": len(processed.split()),
            "note": "Tách từ bằng Underthesea, từ ghép nối bằng dấu _." if spec["segmented"]
            else "Cấu hình A/B không dùng Underthesea; giữ văn bản đã làm sạch như lúc huấn luyện."})

        yield {"type": "stage", "stage": "tfidf", "status": "running"}
        start = perf_counter()
        model = self.load(identifier)
        vectorizer, svc = model.named_steps["tfidf"], model.named_steps["svc"]
        vector = vectorizer.transform([processed]).tocsr()
        counts = Counter(vectorizer.build_analyzer()(processed))
        names = vectorizer.get_feature_names_out()
        features = [{"term": str(names[index]), "index": int(index), "tf": counts[str(names[index])],
                     "idf": float(vectorizer.idf_[index]), "weight": float(weight)}
                    for index, weight in zip(vector.indices, vector.data)]
        features.sort(key=lambda item: (-item["weight"], item["index"]))
        oov = [{"term": term, "count": count} for term, count in counts.items() if term not in vectorizer.vocabulary_]
        oov.sort(key=lambda item: (-item["count"], item["term"]))
        if not vector.nnz:
            raise InputError("Không có từ nào khớp từ vựng TF-IDF đã học. Không đủ đặc trưng để đưa ra dự đoán có ý nghĩa.")
        if len(counts) < 10:
            warnings.append("Nội dung rất ngắn; mô hình có ít ngữ cảnh để dự đoán.")
        if len(oov) > len(counts) / 2:
            warnings.append("Hơn một nửa số từ khác nhau không có trong từ vựng đã học; cần thận trọng với kết quả.")
        yield finished("tfidf", start, {"features": features, "vocabulary_size": len(names),
            "active_features": int(vector.nnz), "analyzed_terms": sum(counts.values()),
            "unique_terms": len(counts), "oov": oov, "norm": vectorizer.norm,
            "l2_norm": float(np.sqrt(vector.multiply(vector).sum())),
            "formula": "TF = số lần xuất hiện; IDF = log((1 + N) / (1 + df)) + 1; trọng số = TF × IDF, chuẩn hóa L2."})

        yield {"type": "stage", "stage": "svm", "status": "running"}
        start = perf_counter()
        # Predict with SVC itself (including its exact boundary/tie behavior).
        label = int(svc.predict(vector)[0])
        score = float(svc.decision_function(vector)[0])
        bias = float(svc.intercept_[0])
        if svc.kernel == "linear":
            coefficients = svc.coef_.toarray().ravel()
            terms = [{"term": feature["term"], "weight": feature["weight"],
                      "coefficient": float(coefficients[feature["index"]]),
                      "contribution": float(feature["weight"] * coefficients[feature["index"]])}
                     for feature in features]
            terms.sort(key=lambda item: -abs(item["contribution"]))
            explanation = {"kind": "linear", "formula": "f(x) = Σ wⱼ × xⱼ + b", "terms": terms,
                           "contribution_sum": sum(item["contribution"] for item in terms)}
        else:
            similarities = rbf_kernel(vector, svc.support_vectors_, gamma=svc._gamma).ravel()
            dual = svc.dual_coef_.toarray().ravel()
            contributions = similarities * dual
            top = np.argsort(-np.abs(contributions), kind="stable")[:20]
            explanation = {"kind": "rbf", "formula": "f(x) = Σ αᵢyᵢ × exp(−γ ‖x − svᵢ‖²) + b",
                "gamma": float(svc._gamma), "contribution_sum": float(contributions.sum()),
                "top_support_vectors": [{"index": int(index), "similarity": float(similarities[index]),
                    "dual_coefficient": float(dual[index]), "contribution": float(contributions[index])} for index in top],
                "note": "TF-IDF mô tả văn bản đầu vào, không phải mức ảnh hưởng của từng từ trong SVM RBF. Bảng hiển thị tối đa 20 vector hỗ trợ có |đóng góp| lớn nhất; tổng được tính trên tất cả vector hỗ trợ."}
        result = {"label": label, "label_name": LABELS[label], "decision_score": score,
            "threshold": 0, "bias": bias, "explanation": explanation,
            "support_vectors": int(svc.n_support_.sum()), "support_vectors_by_class": svc.n_support_.tolist(),
            "parameters": {"kernel": svc.kernel, "C": svc.C, "gamma": svc.gamma,
                           "class_weight": svc.class_weight},
            "model": {**{k: v for k, v in spec.items() if k not in {"path", "sha256"}},
                      "artifact": spec["path"].relative_to(self.root).as_posix()},
            "warnings": warnings, "disclaimer": DISCLAIMER}
        yield finished("svm", start, result)
        yield {"type": "complete", "duration_ms": round((perf_counter() - overall_start) * 1000, 3),
               "result": result, "stages": stages}
