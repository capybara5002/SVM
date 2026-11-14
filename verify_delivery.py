"""Verify numbers, ID/fold provenance, figures and local report links."""
import argparse
from html.parser import HTMLParser
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw
import pymupdf
from sklearn.metrics import roc_auc_score
from src.config import BASE_DIR
from src.evaluate_svm import compute_metrics, load_processed_data
from src.experiment_abcd import audit_inputs


class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.links = []
    def handle_starttag(self,tag,attrs):
        self.links.extend(v for k,v in attrs if k in ("href","src"))


def check(staging=None):
    docs = BASE_DIR / "docs"
    target = BASE_DIR / staging if staging else docs
    root = BASE_DIR / "results/chapter3_4"
    summary = json.loads((root / "experiment_summary.json").read_text(encoding="utf-8"))
    raw = pd.read_csv(BASE_DIR / "data/raw/reintel_train.csv")
    train = load_processed_data(BASE_DIR / "data/processed/train.csv","train")
    test = load_processed_data(docs / "tables/evaluated_test_samples.csv","test")
    _,_,_,audit = audit_inputs(train,test,raw,pd.read_csv(root / "cv_fold_assignments.csv"))
    expected_labels = test.set_index("source_index").label
    table = pd.read_csv(root / "metrics_abcd.csv").set_index("scenario")
    checked_models = []
    for model in summary["models"]:
        code = model["scenario"]
        predictions = pd.read_csv(root / f"predictions_{code}.csv")
        assert predictions.source_index.tolist() == test.source_index.tolist(), code
        assert np.array_equal(expected_labels.loc[predictions.source_index],predictions.y_true), code
        assert np.array_equal((predictions.decision_score > 0).astype(int),predictions.y_pred), code
        result = compute_metrics(predictions.y_true,predictions.y_pred)
        result["metrics"]["roc_auc"] = roc_auc_score(predictions.y_true,predictions.decision_score)
        assert result["confusion_matrix"] == model["confusion_matrix"],code
        assert np.asarray(model["confusion_matrix"]).sum() == 856
        assert np.array_equal(np.asarray(model["confusion_matrix"]).sum(axis=1),[711,145])
        for key,value in result["metrics"].items():
            assert np.isclose(value,model["metrics"][key],rtol=0,atol=1e-12),(code,key)
            assert np.isclose(value,table.loc[code,key],rtol=0,atol=1e-12),(code,key)
        assert np.array_equal(np.asarray(model["confusion_matrix"]).ravel(),table.loc[code,["tn","fp","fn","tp"]].to_numpy())
        grid = pd.read_csv(root / f"grid_search_{code}.csv")
        assert len(grid) == (8 if model["kernel"] == "linear" else 32)
        assert model["cv_fits"] == 5 * len(grid)
        index = int(grid.mean_test_f1.argmax())
        assert json.loads(grid.loc[index,"params"]) == model["params"]
        folds = np.array([grid.loc[index,f"split{i}_test_f1"] for i in range(5)])
        assert np.isclose(folds.mean(),model["cv_f1_mean"],atol=1e-12)
        assert np.isclose(folds.std(),model["cv_f1_std"],atol=1e-12)
        assert hashlib.sha256((BASE_DIR/model["model_file"]).read_bytes()).hexdigest() == model["model_sha256"]
        benchmark = model["benchmark"]
        assert (benchmark["samples"],benchmark["batch_size"],benchmark["repeats"]) == (128,64,5)
        checked_models.append(code)
    assert max(summary["models"],key=lambda m:m["cv_f1_mean"])["scenario"] == summary["selected_by_cv"]
    assert sum(m["cv_fits"] for m in summary["models"]) == 400
    for key,path in {"raw":"data/raw/reintel_train.csv","train":"data/processed/train.csv","test":"data/processed/test.csv", "filtered_test":"docs/tables/evaluated_test_samples.csv","folds":"results/week4_5/cv_fold_assignments.csv"}.items():
        assert hashlib.sha256((BASE_DIR/path).read_bytes()).hexdigest() == summary["input_sha256"][key],key
    legacy = json.loads((docs / "evaluation_summary.json").read_text(encoding="utf-8"))
    for model in legacy["models"]:
        predictions = pd.read_csv(docs / model["prediction_file"])
        assert predictions.source_index.tolist() == test.source_index.tolist()
        assert compute_metrics(predictions.y_true,predictions.y_pred)["confusion_matrix"] == model["confusion_matrix"]
    # Reused Underthesea C/D must reproduce the old tuned predictions and CV.
    for code,legacy_id in (("C","svm_linear_tuned"),("D","svm_rbf_tuned")):
        old = next(m for m in legacy["models"] if m["id"] == legacy_id)
        new = next(m for m in summary["models"] if m["scenario"] == code)
        assert old["confusion_matrix"] == new["confusion_matrix"]
        assert np.isclose(old["cv_f1"],new["cv_f1_mean"],atol=1e-12)
        legacy_predictions = pd.read_csv(docs / old["prediction_file"])
        new_predictions = pd.read_csv(root / f"predictions_{code}.csv")
        assert np.array_equal(legacy_predictions.y_pred,new_predictions.y_pred)
        assert np.allclose(legacy_predictions.decision_score,new_predictions.decision_score,rtol=0,atol=1e-10)
    image_checks = []
    svg_previews = BASE_DIR / "tmp/svg_review"
    svg_previews.mkdir(parents=True,exist_ok=True)
    for path in sorted((target / "figures").glob("*.png")):
        svg = path.with_suffix(".svg")
        assert svg.is_file(),path
        ET.parse(svg)
        with Image.open(path) as im:
            assert min(im.info.get("dpi",(0,0))) >= 299.9,path
            assert im.convert("RGBA").getchannel("A").getextrema() == (255,255),path
            assert im.convert("RGB").getpixel((0,0)) == (255,255,255),path
            dimensions = im.size
            if path.stem == "abcd_slide_16x9":
                assert dimensions[0]*9 == dimensions[1]*16
        vector = pymupdf.open(svg)
        vector[0].get_pixmap(matrix=pymupdf.Matrix(1.3,1.3),alpha=False).save(svg_previews / path.name)
        vector.close()
        with Image.open(svg_previews/path.name) as rendered,Image.open(path) as raster:
            raster_array=np.asarray(raster.convert("RGB").resize(rendered.size,Image.Resampling.LANCZOS),dtype=float)
            svg_array=np.asarray(rendered.convert("RGB"),dtype=float)
            mean_render_difference=float(np.abs(raster_array-svg_array).mean()/255)
            assert mean_render_difference < .02,(path,mean_render_difference)
        manifest = json.loads(path.with_suffix(".export.json").read_text(encoding="utf-8"))
        assert not manifest["clipped_text"],path
        assert hashlib.sha256((BASE_DIR/manifest["source_data"]).read_bytes()).hexdigest() == manifest["source_sha256"]
        for ext in ("png","svg"):
            assert hashlib.sha256(path.with_suffix("."+ext).read_bytes()).hexdigest() == manifest["sha256"][ext]
        image_checks.append({"name":path.stem,"dimensions":dimensions,"dpi":300,"svg_rendered":True,
                             "png_svg_mean_absolute_difference":mean_render_difference})
    missing,link_count = [],0
    for filename in ("index.html","README.md","chapter3_4.html","chuong_3_4.md"):
        content = (target / filename).read_text(encoding="utf-8")
        if filename.endswith("html"):
            parser = Links(); parser.feed(content); links=parser.links
        else:
            links = re.findall(r"\]\(([^)]+)\)",content)
        for url in links:
            if not url or url.startswith("#") or urlsplit(url).scheme:
                continue
            relative = unquote(url.split("#")[0])
            # Staged docs resolve their unchanged tables/model links in the final docs location.
            candidate = target / relative
            if not candidate.exists(): candidate = docs / relative
            if not candidate.exists(): missing.append((filename,url))
            link_count += 1
    assert not missing,missing
    if not staging:
        for url in re.findall(r"\]\(([^)]+)\)",(BASE_DIR/"README.md").read_text(encoding="utf-8")):
            if not urlsplit(url).scheme and not url.startswith("#"):
                assert (BASE_DIR/unquote(url.split("#")[0])).exists(),url
                link_count += 1
    annotations = pd.read_csv(root / "error_annotations.csv")
    examples = pd.read_csv(root / "error_examples.csv")
    assert set(examples.source_index).issubset(set(annotations.source_index))
    report = {"passed":True,"models":checked_models,"train_rows":audit["train_rows"],"test_rows":audit["test_rows"],
              "matrices_checked":8,"figures":image_checks,"local_links_checked":link_count,"missing_links":missing,
              "note":"Numerical, metadata, SVG-render and boundary checks; manual visual review is separate"}
    if staging:
        report["staging_manifest_sha256"] = hashlib.sha256((target/"staging_manifest.json").read_bytes()).hexdigest()
    (target / "qa_checks.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"PASS: 4 scenarios + 4 legacy models; {len(image_checks)} PNG/SVG pairs; {link_count} local links")
    return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--staging",type=Path)
    args=parser.parse_args();check(args.staging)
