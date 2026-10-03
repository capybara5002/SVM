"""Chay huan luyen SVM va Grid Search theo ke hoach tuan 4-5."""

from __future__ import annotations

import argparse
import sys

from src.train_svm import train_models


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    parser = argparse.ArgumentParser(description="Huấn luyện SVM Linear/RBF với 5-fold CV; chỉ dùng train.")
    parser.add_argument("--n-jobs", type=int, default=2, help="Số tác vụ song song (mặc định: 2).")
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=1, help="Mức chi tiết tiến độ Grid Search.")
    args = parser.parse_args()
    if args.n_jobs < 1:
        parser.error("--n-jobs phải là số nguyên dương.")
    try:
        train_models(n_jobs=args.n_jobs, verbose=args.verbose)
    except (FileNotFoundError, ValueError) as error:
        print(f"LỖI HUẤN LUYỆN: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
