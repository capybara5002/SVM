"""Chia tap huan luyen/kiem thu tai lap duoc."""

from __future__ import annotations

import warnings
from math import ceil

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import RANDOM_STATE, TEST_SIZE


def _can_stratify(labels: pd.Series, test_size: float) -> bool:
    counts = labels.value_counts(dropna=False)
    class_count = len(counts)
    test_count = ceil(len(labels) * test_size)
    train_count = len(labels) - test_count
    return bool(
        class_count >= 2
        and counts.min() >= 2
        and test_count >= class_count
        and train_count >= class_count
    )


def split_dataframe(
    dataframe: pd.DataFrame,
    label_column: str,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chia 80/20 va phan tang theo nhan neu du lieu cho phep."""
    if len(dataframe) < 2:
        raise ValueError("Cần ít nhất 2 mẫu để chia tập huấn luyện và kiểm thử.")
    stratify = dataframe[label_column] if _can_stratify(dataframe[label_column], test_size) else None
    if stratify is None:
        warnings.warn(
            "Không đủ mẫu để phân tầng theo nhãn; đang chia ngẫu nhiên có random_state.",
            UserWarning,
            stacklevel=2,
        )
    train, test = train_test_split(
        dataframe,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return train.copy(), test.copy()
