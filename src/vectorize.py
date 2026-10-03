"""Bieu dien van ban bang TF-IDF ma khong ro ri du lieu."""

from __future__ import annotations

from collections.abc import Iterable

from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import MAX_FEATURES, NGRAM_RANGE


def create_tfidf_vectorizer(
    max_features: int = MAX_FEATURES,
    ngram_range: tuple[int, int] = NGRAM_RANGE,
) -> TfidfVectorizer:
    """Dung chung cau hinh TF-IDF cho chuan bi du lieu va tung fold CV."""
    return TfidfVectorizer(
        lowercase=False,
        max_features=max_features,
        ngram_range=ngram_range,
        token_pattern=r"(?u)\b\w+\b",
    )


def fit_tfidf_on_train(
    train_texts: Iterable[str],
    test_texts: Iterable[str],
    max_features: int = MAX_FEATURES,
    ngram_range: tuple[int, int] = NGRAM_RANGE,
) -> tuple[TfidfVectorizer, spmatrix, spmatrix]:
    """Chi fit tren train; test chi duoc transform bang tu vung da hoc."""
    vectorizer = create_tfidf_vectorizer(max_features, ngram_range)
    try:
        train_matrix = vectorizer.fit_transform(train_texts)
    except ValueError as error:
        raise ValueError(
            "Không tạo được từ vựng TF-IDF từ tập huấn luyện. "
            "Hãy kiểm tra văn bản sau tiền xử lý."
        ) from error
    test_matrix = vectorizer.transform(test_texts)
    return vectorizer, train_matrix, test_matrix
