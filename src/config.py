"""Cau hinh trung tam cho chuan bi du lieu, huan luyen va danh gia SVM."""

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

CV_FOLDS = 5
SVM_C_VALUES = (0.1, 1.0, 10.0, 100.0)
SVM_GAMMA_VALUES = ("scale", 0.01, 0.1, 1.0)
SVM_CLASS_WEIGHTS = (None, "balanced")
TRAINING_RESULTS_DIR = RESULTS_DIR / "week4_5"
DOCS_DIR = BASE_DIR / "docs"

SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".txt", ".json", ".jsonl", ".zip"}
