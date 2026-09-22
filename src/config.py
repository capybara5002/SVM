"""Cau hinh trung tam cho pipeline chuan bi du lieu tuan 1-3."""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
RESULTS_DIR = BASE_DIR / "results"

# Neu data/raw chi co mot tep du lieu, co the de None de tu dong tim.
# Neu co nhieu tep, hay dien ten tep, vi du: "ReINTEL.csv".
DATA_FILE: str | None = "reintel_train.csv"

# De None de tu nhan dien theo cac ten cot pho bien. Neu chuong trinh bao
# khong chac chan, hay dien CHINH XAC ten cot trong bo du lieu that.
TEXT_COLUMN: str | None = "text"
LABEL_COLUMN: str | None = "label"

RANDOM_STATE = 42
TEST_SIZE = 0.20
MAX_FEATURES = 10_000
NGRAM_RANGE = (1, 1)

SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".txt", ".json", ".jsonl", ".zip"}
