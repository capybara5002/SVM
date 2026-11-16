"""Local Flask app serving the SVM lab and streaming actual inference stages."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from flask import Flask, Response, jsonify, request, stream_with_context
from werkzeug.exceptions import RequestEntityTooLarge

from src.web_analysis import AnalysisEngine, EXTENSIONS, InputError, MAX_FILE_BYTES, validate_text


def create_app(engine: AnalysisEngine | None = None) -> Flask:
    app = Flask(__name__, static_folder="web/static", template_folder="web")
    app.config.update(MAX_CONTENT_LENGTH=MAX_FILE_BYTES + 128 * 1024, MAX_FORM_MEMORY_SIZE=512 * 1024)
    engine = engine or AnalysisEngine()
    app.extensions["analysis_engine"] = engine

    @app.after_request
    def headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; style-src 'self'; script-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
            "base-uri 'self'; frame-ancestors 'none'; form-action 'self'"
        )
        return response

    @app.get("/")
    def index():
        return Response(
            (Path(app.root_path) / "web/index.html").read_text(encoding="utf-8"), mimetype="text/html")

    @app.get("/api/models")
    def models():
        return jsonify(engine.catalog())

    @app.errorhandler(RequestEntityTooLarge)
    def oversized(_error):
        return jsonify(error="Tệp quá lớn. Giới hạn tải lên là 5 MB."), 413

    @app.post("/api/analyze")
    def analyze():
        # Local uploads must originate from this app, including on a loopback host.
        if request.headers.get("Origin") and request.headers["Origin"] != request.host_url.rstrip("/"):
            return jsonify(error="Yêu cầu phải được gửi từ trang phân tích này."), 403
        if not engine.specs:
            return jsonify(error="Chưa có mô hình. Hãy tạo artifacts bằng quy trình huấn luyện trong README."), 503
        identifier = request.form.get("model", engine.default)
        if identifier not in engine.specs:
            return jsonify(error="Mô hình không khả dụng. Hãy tải lại trang."), 400
        mode = request.form.get("mode", "text")
        text, filename, content = "", None, b""
        try:
            if mode == "file":
                upload = request.files.get("file")
                if upload is None or not upload.filename:
                    raise InputError("Hãy chọn tệp cần phân tích.")
                filename = upload.filename.replace("\\", "/").rsplit("/", 1)[-1]
                if Path(filename).suffix.lower() not in EXTENSIONS:
                    raise InputError("Chỉ hỗ trợ .txt, .md, .docx và .pdf có lớp văn bản.")
                content = upload.read(MAX_FILE_BYTES + 1)
                if len(content) > MAX_FILE_BYTES:
                    raise InputError("Tệp vượt quá giới hạn 5 MB.")
            elif mode == "text":
                text = validate_text(request.form.get("text", ""))
            else:
                raise InputError("Chế độ nhập không hợp lệ.")
        except InputError as error:
            return jsonify(error=str(error)), 400

        def generate():
            try:
                for event in engine.events(identifier, text, filename, content):
                    yield json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n"
            except InputError as error:
                yield json.dumps({"type": "error", "error": str(error)}, ensure_ascii=False) + "\n"
            except Exception:
                logging.exception("Analysis failed")
                yield json.dumps({"type": "error", "error": "Không thể phân tích. Hãy kiểm tra mô hình và thư viện trong cửa sổ chạy server."}, ensure_ascii=False) + "\n"

        return Response(stream_with_context(generate()), mimetype="application/x-ndjson",
                        headers={"X-Accel-Buffering": "no"})

    return app
