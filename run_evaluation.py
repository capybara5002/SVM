"""Chay danh gia test va sinh bang, bieu do trong thu muc docs."""

import argparse
import sys

from src.evaluate_svm import evaluate_saved_models


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    parser = argparse.ArgumentParser(description="Đánh giá 4 cấu hình SVM đã lưu và tạo báo cáo docs/.")
    parser.add_argument("--no-render", action="store_true", help="Chỉ xuất số liệu; dùng run_visuals.py để kiểm tra ảnh tạm trước khi thay.")
    args = parser.parse_args()
    try:
        evaluate_saved_models(render_docs=not args.no_render)
    except (FileNotFoundError, ValueError, KeyError) as error:
        print(f"LỖI ĐÁNH GIÁ: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
