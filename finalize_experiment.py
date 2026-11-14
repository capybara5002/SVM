"""Persist lightweight data statistics and reproducibility records after training."""
import hashlib
import json
from pathlib import Path
import pandas as pd
from src.config import BASE_DIR
from src.experiment_abcd import write_csv, write_json
from src.preprocess import clean_text
from run_preparation import _prepare_valid_rows


def finalize():
    root = BASE_DIR / "results/chapter3_4"
    raw = pd.read_csv(BASE_DIR / "data/raw/reintel_train.csv")
    train = pd.read_csv(BASE_DIR / "data/processed/train.csv")
    test = pd.read_csv(BASE_DIR / "docs/tables/evaluated_test_samples.csv")
    write_csv(test[["source_index", "label"]].head(128),root / "benchmark_ids.csv")
    valid = _prepare_valid_rows(raw,"text","label")
    clean_valid = valid.text.map(clean_text)
    selected_ids = set(valid.loc[clean_valid.ne(""),"source_index"])
    original_test = pd.read_csv(BASE_DIR / "data/processed/test.csv")
    assert selected_ids == set(train.source_index) | set(original_test.source_index)
    missing_or_empty = raw.text.isna() | raw.text.fillna("").str.strip().eq("") | raw.label.isna()
    write_json({"raw_rows":len(raw),"missing_or_empty":int(missing_or_empty.sum()),
        "raw_duplicate_rows":len(raw)-int(missing_or_empty.sum())-len(valid),"valid_rows":len(valid),
        "empty_after_cleaning":int(clean_valid.eq("").sum()),"rows_before_split":len(selected_ids)},root/"preparation_audit.json")
    rows=[]
    for name,frame,ids in (("raw",raw,raw.index),("train",train,train.source_index),("holdout",test,test.source_index)):
        texts=raw.loc[ids,"text"].dropna().astype(str)
        cleaned=texts.map(clean_text)
        rows.append({"split":name,"rows":len(frame),"nonmissing_text_rows":len(texts),"label_0":int(frame.label.eq(0).sum()),"label_1":int(frame.label.eq(1).sum()),
            "raw_chars_mean":float(texts.str.len().mean()),"raw_chars_median":float(texts.str.len().median()),
            "clean_chars_mean":float(cleaned.str.len().mean()),"clean_whitespace_units_mean":float(cleaned.str.split().str.len().mean())})
    write_csv(pd.DataFrame(rows),root/"dataset_statistics.csv")
    legacy = pd.read_csv(BASE_DIR / "docs/tables/model_metrics.csv")
    write_csv(legacy,root/"legacy_reference_metrics.csv")
    source_paths = ["src/experiment_abcd.py","src/preprocess.py","src/vectorize.py","src/train_svm.py","src/config.py",
                    "src/evaluation_visuals.py","src/chapter_report.py","run_experiments.py","run_visuals.py","verify_delivery.py","requirements.txt"]
    hashes={path:hashlib.sha256((BASE_DIR/path).read_bytes()).hexdigest() for path in source_paths}
    skill=Path("C:/Users/jiki/.codex/skills/scientific-visualization/SKILL.md")
    write_json({"source_sha256":hashes,"skill":{"path":str(skill),"sha256":hashlib.sha256(skill.read_bytes()).hexdigest(),
        "repo":"https://github.com/K-Dense-AI/scientific-agent-skills","path_in_repo":"skills/scientific-visualization",
        "installation":"Attempted requested npx.cmd command from DACN; clone stalled. Installed only this skill and its 17 support files through GitHub API/raw download."}},root/"reproducibility.json")
    print("Persisted preparation audit, statistics, benchmark IDs and provenance")


if __name__ == "__main__":
    finalize()
