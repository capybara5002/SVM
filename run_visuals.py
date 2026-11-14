"""Stage reproducible figures; publish only the files reviewed in staging."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from src.config import BASE_DIR
from src.evaluation_visuals import render_evaluation_docs, render_abcd_figures
from src.chapter_report import write_chapter_report


def stage(output):
    legacy = json.loads((BASE_DIR / "docs/evaluation_summary.json").read_text(encoding="utf-8"))
    abcd = json.loads((BASE_DIR / "results/chapter3_4/experiment_summary.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True,exist_ok=True)
    render_evaluation_docs(legacy,output)
    render_abcd_figures(abcd,output / "figures")
    write_chapter_report(abcd,output)
    for path in (output / "figures").glob("*.export.json"):
        info = json.loads(path.read_text(encoding="utf-8"))
        source = "results/chapter3_4/experiment_summary.json" if info["name"].startswith("abcd_") else "docs/evaluation_summary.json"
        info["source_data"] = source
        info["source_sha256"] = hashlib.sha256((BASE_DIR/source).read_bytes()).hexdigest()
        info["transformations"] = "metrics unchanged; normalized confusion cells divide by true-label row total"
        path.write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding="utf-8")
    for file in ("index.html", "README.md"):
        path = output / file
        text = path.read_text(encoding="utf-8")
        if file.endswith("html"):
            text = text.replace("<nav aria-label=",'<p class="note"><a href="chapter3_4.html"><strong>Thực nghiệm A–D và nội dung Chương 3–4</strong></a> · <a href="chuong_3_4.md">Bản Markdown</a></p><nav aria-label=')
            text = text.replace("Bốn cấu hình sử dụng cùng tập test; màu xanh ngọc đánh dấu cấu hình chọn theo CV.","Xanh: Linear; cam: RBF; họa tiết: cơ sở; tô đặc: tối ưu; dấu sao: lựa chọn CV.")
        else:
            text = "[Thực nghiệm A–D và nội dung Chương 3–4](chuong_3_4.md) · [Mở HTML A–D](chapter3_4.html)\n\n" + text
        text = text.replace("Test độc lập", "Holdout nội bộ").replace("test độc lập", "holdout nội bộ")
        text = text.replace("Tính độc lập của tập test", "Phạm vi của holdout nội bộ")
        path.write_text(text,encoding="utf-8")
    manifest = {str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in output.rglob("*") if p.is_file() and p.suffix in (".png",".svg",".md",".html")}
    (output / "staging_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    (output / "qa_checks.json").write_text(json.dumps({"passed":False,"reason":"Regenerated; rerun verify_delivery and visually inspect PNG/SVG"}),encoding="utf-8")
    print(f"Staged {len(list((output/'figures').glob('*.png')))} PNG/SVG pairs: {output}")


def publish(output):
    manifest = json.loads((output / "staging_manifest.json").read_text(encoding="utf-8"))
    qa = json.loads((output / "qa_checks.json").read_text(encoding="utf-8"))
    if not qa["passed"]:
        raise ValueError("Staging checks have not passed")
    if qa.get("staging_manifest_sha256") != hashlib.sha256((output/"staging_manifest.json").read_bytes()).hexdigest():
        raise ValueError("Staging manifest changed after verification")
    for relative, expected in manifest.items():
        if hashlib.sha256((output / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Staging changed since review: {relative}")
    docs = BASE_DIR / "docs"
    # Update all references before replacing files. Every previous basename is retained.
    replaced = [p.name for p in (docs / "figures").glob("*") if p.suffix in (".png",".svg")]
    for filename in ("index.html","README.md","chapter3_4.html","chuong_3_4.md"):
        shutil.copy2(output / filename,docs / filename)
    for path in (output / "figures").iterdir():
        if path.is_file():
            shutil.copy2(path,docs / "figures" / path.name)
    record = {"replaced":replaced,"deleted":[],"added":[p.name for p in (output / "figures").glob("*") if p.suffix in (".png",".svg") and p.name not in replaced],
              "note":"All old basenames retained; no deletion required", "staging_manifest_sha256":hashlib.sha256((output/'staging_manifest.json').read_bytes()).hexdigest()}
    (docs / "figure_changes.json").write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Published {len(replaced)} replacements, {len(record['added'])} additions; deleted 0 files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=BASE_DIR / "tmp/visuals_review")
    parser.add_argument("--publish",action="store_true")
    args = parser.parse_args()
    if args.publish:
        publish(args.output)
    else:
        stage(args.output)
