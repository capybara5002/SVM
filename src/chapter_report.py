"""Generate report-ready Vietnamese Chapter 3/4 evidence from persisted results."""
from __future__ import annotations
import json
import re
from html import escape
from pathlib import Path
import pandas as pd
from .config import BASE_DIR


def table(headers, rows):
    return "| " + " | ".join(headers) + " |\n|" + "|".join(["---"]*len(headers)) + "|\n" + "\n".join("| " + " | ".join(map(str,r)) + " |" for r in rows)


def _inline(text):
    value = escape(text)
    value = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)",r'<figure><img alt="\1" src="\2"><figcaption>\1</figcaption></figure>',value)
    value = re.sub(r"\[([^\]]*)\]\(([^)]+)\)",r'<a href="\2">\1</a>',value)
    value = re.sub(r"`([^`]+)`",r"<code>\1</code>",value)
    value = re.sub(r"\*\*([^*]+)\*\*",r"<strong>\1</strong>",value)
    return value


def markdown_html(markdown):
    """Small offline renderer for the headings/paragraphs/tables used here."""
    lines, output, i = markdown.splitlines(), [], 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith("```"):
            block = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(lines[i]); i += 1
            output.append("<pre><code>" + escape("\n".join(block)) + "</code></pre>")
            i += 1
            continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([x.strip() for x in lines[i].strip("|").split("|")]); i += 1
            output.append("<div class='scroll'><table><thead><tr>" + "".join("<th>"+_inline(x)+"</th>" for x in rows[0]) + "</tr></thead><tbody>" + "".join("<tr>"+"".join("<td>"+_inline(x)+"</td>" for x in row)+"</tr>" for row in rows[2:]) + "</tbody></table></div>")
            continue
        match = re.match(r"^(#{1,6}) (.*)",line)
        if match:
            n = len(match[1]); output.append(f"<h{n}>"+_inline(match[2])+f"</h{n}>")
        elif line.startswith("!["):
            output.append(_inline(line))
        else:
            output.append("<p>"+_inline(line)+"</p>")
        i += 1
    return "\n".join(output)


def write_chapter_report(summary: dict, docs_dir: Path):
    root = BASE_DIR / "results/chapter3_4"
    metrics = pd.read_csv(root / "metrics_abcd.csv")
    pairs = pd.read_csv(root / "pairwise_comparisons.csv")
    sensitivity = pd.read_csv(root / "near_duplicate_sensitivity.csv")
    examples = pd.read_csv(root / "error_examples.csv")
    data_stats = pd.read_csv(root / "dataset_statistics.csv")
    stats_table = table(["Tập","Số mẫu","Nhãn 0","Nhãn 1","Ký tự gốc TB","Ký tự sạch TB","Đơn vị khoảng trắng TB"],
        [[r['split'],int(r.rows),int(r.label_0),int(r.label_1),f"{r.raw_chars_mean:.1f}",f"{r.clean_chars_mean:.1f}",f"{r.clean_whitespace_units_mean:.1f}"] for _,r in data_stats.iterrows()])
    annotation_path = root / "error_annotations.csv"
    annotations = pd.read_csv(annotation_path) if annotation_path.exists() else pd.DataFrame(columns=["source_index","comment"])
    annotation_map = annotations.set_index("source_index")["comment"].to_dict()
    selected = summary["selected_by_cv"]
    winner = next(m for m in summary["models"] if m["scenario"] == selected)
    pct = lambda x: f"{100*x:.2f}%"
    report_rows = [[m["scenario"],"Có" if m["underthesea"] else "Không",m["kernel"],
        "; ".join(f"{k.removeprefix('svc__')}={v}" for k,v in m["params"].items()),
        f"{m['cv_f1_mean']:.4f} ± {m['cv_f1_std']:.4f}",
        *[f"{m['metrics'][k]:.4f}" for k in ("accuracy","precision","recall","f1","roc_auc")],
        "/".join(map(str,[*m["confusion_matrix"][0],*m["confusion_matrix"][1]]))] for m in summary["models"]]
    scores_table = table(["Kịch bản","UTS","Kernel","Tham số chọn trên CV","F1 CV ± SD","Accuracy","Precision₁","Recall₁","F1₁","ROC-AUC","TN/FP/FN/TP"],report_rows)
    fold_table = table(["Fold","Train 0","Train 1","Validation 0","Validation 1","Tổng validation"],
        [[r[k] for k in ("fold","train_0","train_1","validation_0","validation_1","validation_n")] for r in summary["audit"]["fold_details"]])
    paired_table = table(["Cặp (sau − trước)","Yếu tố","Δ F1 CV","Δ Accuracy","Δ Precision₁","Δ Recall₁","Δ F1₁","Δ AUC"],
        [[r["pair"],r["factor"],*[f"{r[k+'_delta_pp']:+.2f}" for k in ("cv_f1_mean","accuracy","precision","recall","f1","roc_auc")]] for _,r in pairs.iterrows()])
    time_table = table(["Kịch bản","Grid Search (s)","Refit (s)","Pipeline (ms/mẫu)","SD (ms/mẫu)","Toàn luồng (ms/mẫu)","Tệp (MiB)"],
        [[m["scenario"],f"{m['grid_search_seconds']:.3f}",f"{m['refit_seconds']:.3f}",
          f"{m['benchmark']['pipeline_median_ms_per_sample']:.3f}",f"{m['benchmark']['pipeline_std_ms_per_sample']:.3f}",
          f"{m['benchmark']['end_to_end_median_ms_per_sample']:.3f}",f"{m['model_bytes']/2**20:.3f}"] for m in summary["models"]])
    pre_table = table(["Nhánh","Tiền xử lý 3.428 train (s)","Tiền xử lý 856 test (s)"],
        [[name,f"{r['train_seconds']:.3f}",f"{r['test_seconds']:.3f}"] for name,r in summary["preprocessing"].items()])
    sensitivity_table = table(["Ngưỡng cosine ≥","Kịch bản","Loại","Giữ","Số nhãn 1","F1₁","Recall₁","ROC-AUC"],
        [[r.threshold,r.scenario,int(r.excluded),int(r.retained),int(r.positive_support),*[f"{r[k]:.4f}" for k in ("f1","recall","roc_auc")]] for _,r in sensitivity.iterrows()])
    strict = sensitivity.loc[(sensitivity.threshold == .8) & (sensitivity.scenario == selected)].iloc[0]
    near_note = f"Ở ngưỡng 0.80, giữ {int(strict.retained)} mẫu, loại {int(strict.excluded)}; F1 nhãn 1 của cấu hình CV {selected} là {strict.f1:.4f}, thay đổi {100*(strict.f1-winner['metrics']['f1']):+.2f} điểm phần trăm so với holdout đầy đủ."
    error_sections = []
    for _, row in examples.iterrows():
        excerpt = str(row.text_original).replace("\n"," ").replace("\r"," ").strip()
        if len(excerpt) > 1100:
            excerpt = excerpt[:1100] + " […]"
        comment = annotation_map.get(row.source_index,"Chưa có nhận xét thủ công; cần đọc nội dung gốc trong CSV.")
        error_sections.append(f"### {row.outcome} · ID {row.source_index} · {'sát' if row.region == 'near_threshold' else 'xa'} ngưỡng\n\nNhãn thật {row.y_true}; dự đoán {row.y_pred}; decision_function = {row.decision_score:+.6f}.\n\nNội dung (trích, có thể rút gọn): {excerpt}\n\nNhận xét: {comment}")
    hardware = summary["hardware"]
    legacy = json.loads((BASE_DIR / "docs/evaluation_summary.json").read_text(encoding="utf-8"))
    legacy_table = table(["Mốc có Underthesea","F1 CV","Accuracy","Precision₁","Recall₁","F1₁"],
        [[m["display_name"],f"{m['cv_f1']:.4f}",*[f"{m['metrics'][k]:.4f}" for k in ("accuracy","precision","recall","f1")]] for m in legacy["models"]])
    pair_paragraphs = []
    for _,r in pairs.iterrows():
        pair_paragraphs.append(f"{r['pair']}: F1 nhãn 1 thay đổi {r.f1_delta_pp:+.2f} điểm phần trăm; Recall {r.recall_delta_pp:+.2f}; ROC-AUC {r.roc_auc_delta_pp:+.2f}. Đây là đối chiếu các cấu hình được tối ưu riêng theo cùng giao thức, không cô lập riêng tác động của tokenizer ở một bộ siêu tham số cố định.")
    content = f"""# Chương 3–4 · Thực nghiệm SVM–TF-IDF trên ReINTEL

Kết quả thực tế: {summary['completed_at']}. [Mở báo cáo HTML](chapter3_4.html) · [Mốc cơ sở/tối ưu có Underthesea](index.html) · [CSV A–D](../results/chapter3_4/metrics_abcd.csv).

## 3.1. Dữ liệu và phạm vi thực nghiệm

Nguồn: ReliableAI/ReINTEL, bản dữ liệu và revision đã ghi trong [metadata](../data/raw/reintel_metadata.json). CSV thô có 4.372 bài đăng. Quy trình chuẩn bị trước đó loại dòng thiếu/rỗng hoặc trùng cặp văn bản–nhãn, giữ 4.290 dòng; tiếp tục loại 4 văn bản rỗng sau làm sạch, còn 4.286 bản ghi để chia 3.428 train và 858 test. Lần đánh giá trước loại 2 mẫu test trùng văn bản sau phân đoạn, giữ 856 mẫu (711 nhãn 0; 145 nhãn 1). Lần này giữ nguyên các ID đó, không lấy mẫu lại. [Danh sách test](../results/chapter3_4/holdout_ids.csv), [ID và fold train](../results/chapter3_4/cv_fold_assignments.csv), [kiểm tra dữ liệu](../results/chapter3_4/data_audit.json) và hash đầu vào trong [JSON thực nghiệm](../results/chapter3_4/experiment_summary.json) cho phép đối chiếu.

{stats_table}

Độ dài đo bằng ký tự của chuỗi Python, bỏ văn bản thiếu khi tính trung bình (raw có 4.371 văn bản không thiếu); đơn vị khoảng trắng là số phần tử sau split trên văn bản sạch, không phải số từ ngôn ngữ học hoặc token của TF-IDF. source_index là vị trí dòng từ 0 trong CSV gốc của dự án, không khẳng định là ID bài đăng nguyên thủy của ReINTEL. [CSV thống kê](../results/chapter3_4/dataset_statistics.csv) và [kiểm tra các bước chuẩn bị](../results/chapter3_4/preparation_audit.json).

Nhãn 0 = reliable (đáng tin cậy), nhãn 1 = unreliable (không đáng tin cậy). Đây là nhãn bài đăng của bộ dữ liệu, không phải kết quả kiểm chứng sự thật do mô hình tạo ra. Train giữ nguyên 4 dòng lặp sau làm sạch và {summary['audit']['train_conflicting_clean_groups']} nhóm có nhãn mâu thuẫn; các nhóm không bị tách qua ranh giới fold. Giữ nguyên nhãn để đối chiếu mốc cũ, không tự sửa nhãn dựa trên dự đoán.

## 3.2. Môi trường và công cụ

CPU: {hardware['cpu']}; {hardware['physical_cpus']} lõi vật lý, {hardware['logical_cpus']} luồng; RAM máy {hardware['ram_bytes']/2**30:.2f} GiB ({hardware['ram_bytes']:,} byte). Hệ điều hành {hardware['os']}. Python {summary['python']}. Grid Search chạy n_jobs={summary['n_jobs']}, SVC cache_size=256 MiB mỗi tác vụ. Đây là ngân sách cache, không phải số đo RAM thực dùng của mô hình.

Phiên bản thực chạy: {json.dumps(summary['versions'],ensure_ascii=False)}. [Phụ thuộc trực tiếp](../requirements.txt) và [phụ thuộc đầy đủ của môi trường](../results/chapter3_4/environment_freeze.txt).

## 3.3. Pipeline và chống rò rỉ

Đối chiếu Bảng 2.1 ở trang in 31 (trang PDF 39) của báo cáo giữa kỳ: A = không Underthesea + Linear; B = không Underthesea + RBF; C = Underthesea + Linear; D = Underthesea + RBF. Các ID và nhãn train/test khớp dữ liệu nguồn. Không có văn bản sau clean_text trùng chính xác giữa train/test, hoặc giữa train/validation trong từng fold. Kiểm tra cũng xác nhận không trùng chuỗi đã phân đoạn qua từng fold. Nhóm gần trùng được khảo sát riêng ở mục 4.4.

Làm sạch thống nhất: chuẩn hóa NFC, giải mã HTML entity, xóa thẻ HTML, bỏ URL, chuyển chữ thường và gom khoảng trắng. Giữ dấu tiếng Việt, phủ định, dấu câu và emoji ở bước cleaning. C/D tiếp tục gọi Underthesea format="text" để nối từ ghép bằng dấu gạch dưới.

Đính chính cách gọi nhánh A/B: TF-IDF dùng analyzer="word", token_pattern=`(?u)\\b\\w+\\b`, lowercase=False. Regex lấy các dãy ký tự Unicode thuộc lớp từ (gồm chữ, số và gạch dưới), giữ token một ký tự; dấu câu làm ranh giới. Ví dụ `học sinh, a 12 học_sinh!` cho token `học`, `sinh`, `a`, `12`, `học_sinh`. A/B không phân đoạn từ ghép tiếng Việt và không phải phép split theo khoảng trắng thuần túy. Dấu câu và emoji không được regex đưa vào đặc trưng dù cleaning giữ chúng.

TF-IDF chung: unigram (1,1), max_features=10.000, min_df=1, max_df=1.0, norm=l2, use_idf=True, smooth_idf=True, sublinear_tf=False, không stopword/stemming. Pipeline TF-IDF → SVC được clone và fit riêng trong mỗi lượt CV; vocabulary và IDF chỉ học từ phần train của fold. Cấu hình đã chọn được refit trên toàn bộ 3.428 train; holdout chỉ transform và dự đoán. SVC dùng probability=False, decision_function có hướng dương về classes_[1]=1; ROC-AUC tính từ điểm liên tục này.

Tái sử dụng nguyên 5 fold StratifiedGroupKFold seed=42 từ mốc cũ; đã kiểm tra khóa nội dung clean_text độc lập với Underthesea hợp lệ trên chính phân hoạch đó. Không tạo lại fold theo từng nhánh. Các fold có kích thước gần nhau và tỷ lệ nhãn không hoàn toàn bằng nhau:

{fold_table}

## 3.4. Grid Search, lựa chọn và refit

Linear: C ∈ {{0.1, 1, 10, 100}}, class_weight ∈ {{None, balanced}}: 8 ứng viên × 5 fold = 40 fit mỗi nhánh. RBF thử thêm gamma ∈ {{scale, 0.01, 0.1, 1}}: 32 ứng viên × 5 fold = 160 fit mỗi nhánh. Tổng A–D: 400 fit CV và 4 refit; không thu hẹp lưới theo kết quả test. Lưới này theo báo cáo giữa kỳ/pipeline hiện có, rộng hơn lưới dự kiến ban đầu trong đề cương.

Chọn tham số bằng F1 nhãn 1 trung bình không trọng số trên 5 validation fold. SD là độ lệch chuẩn quần thể của 5 điểm fold (ddof=0), không phải khoảng tin cậy và các fold không là 5 thí nghiệm độc lập. Khi hòa điểm, chọn ứng viên đầu theo ParameterGrid; khi hòa giữa kịch bản, ưu tiên thứ tự A/B/C/D. [Bản lựa chọn trước holdout](../results/chapter3_4/selection_before_holdout.json) được lưu trước bước tính chỉ số test. Các CSV grid_search_A/B/C/D giữ điểm từng fold của mọi ứng viên.

## 4.1. Kết quả A–D và mốc đối chiếu

**Cấu hình được chọn theo CV: {selected}**, F1 CV = {winner['cv_f1_mean']:.4f} ± {winner['cv_f1_std']:.4f}. **F1 cao nhất trên test (chỉ mô tả): {', '.join(summary['highest_test_f1_descriptive_only'])}**. Không thay cấu hình CV bằng cấu hình thắng test. Accuracy tính trên toàn test; Precision, Recall, F1 dùng pos_label=1, zero_division=0. ROC-AUC dùng decision_function; không dùng nhãn nhị phân thay cho điểm.

{scores_table}

![A–D: Accuracy, Precision, Recall và F1; dấu sao đánh dấu lựa chọn CV](figures/abcd_metric_comparison.png)

Ma trận có hàng là nhãn thật, cột là nhãn dự đoán, thứ tự [0,1]. Từng ma trận tổng 856; mỗi hàng lần lượt tổng 711 và 145. Ma trận chuẩn hóa lấy từng ô chia tổng hàng, không chia tổng 856; mỗi hàng cộng 100%. Các hình cùng loại dùng chung giới hạn thang màu.

![Ma trận A–D: số mẫu](figures/abcd_confusion_overview.png)

![Ma trận A–D: tỷ lệ tính theo hàng](figures/abcd_confusion_overview_normalized.png)

Bốn cấu hình cũ đều có Underthesea, đối chiếu cơ sở và sau Grid Search. Số liệu được giữ làm mốc, không đổi tên chúng thành A–D:

{legacy_table}

![Mốc cũ được vẽ lại: cơ sở có họa tiết, tối ưu tô đặc; xanh Linear, cam RBF](figures/metric_comparison.png)

## 4.2. Ảnh hưởng của Underthesea và kernel

Chênh lệch sau − trước dưới đây dùng đơn vị điểm phần trăm; A→C và B→D thay nhánh phân đoạn, A→B và C→D thay kernel. Dù cách tách token là biến thiết kế, mỗi kịch bản chọn siêu tham số riêng trên CV nên đây là hiệu quả của pipeline đã tối ưu, không phải phép đo tokenizer thuần túy tại C/gamma cố định.

{paired_table}

{chr(10).join(pair_paragraphs)}

Các chênh lệch trên một holdout không đủ để khẳng định ưu thế ổn định hoặc ý nghĩa thống kê. Không kiểm định t-test trên 5 fold có tập train chồng lấn; không gọi SD fold là CI. Cần chia theo sự kiện/nguồn hoặc test ngoài và lặp thực nghiệm để kiểm tra khả năng khái quát.

![Đối chiếu A–C, B–D, A–B và C–D](figures/abcd_pairwise_comparison.png)

## 4.3. Chi phí huấn luyện và suy luận

Đo bằng perf_counter. Tiền xử lý đo lại từ nội dung gốc theo từng nhánh, gồm cleaning và (nếu có) segmentation; không tính I/O, audit ban đầu hoặc việc nạp thư viện. A/B dùng chung phép tiền xử lý không Underthesea; C/D dùng chung phép tiền xử lý Underthesea, không cộng hai lần cùng số đo. Grid Search refit=False, nên thời gian tìm lưới và refit toàn train được đo tách biệt; chi phí kiểm tra/lưu .pkl không nằm trong refit.

{pre_table}

{time_table}

Benchmark suy luận dùng đúng 128 ID đầu của holdout theo thứ tự đã lưu, batch=64 (2 batch/lượt), 5 lần lặp và 1 batch warm-up trước mỗi cấu hình. Pipeline = TF-IDF.transform + SVC.predict trên chuỗi đã tiền xử lý. Toàn luồng = cleaning + segmentation nếu có + pipeline; không bao gồm I/O, nạp mô hình hay decision_function. Giá trị chính là median ms/mẫu của 5 lượt, SD tính từ cùng 5 lượt. [Các ID benchmark](../results/chapter3_4/benchmark_ids.csv) và toàn bộ thời gian từng lượt nằm trong JSON. Mỗi cấu hình dùng cùng văn bản và batching; benchmark chạy tuần tự sau khi hoàn thành mọi Grid Search.

RAM máy là thông số phần cứng. Cột tệp là kích thước .pkl nén joblib compress=3 trên đĩa (MiB = 2²⁰ byte); **chưa đo RAM mô hình sử dụng khi chạy** nên không suy diễn RAM từ kích thước tệp. Thời gian là phép đo trên một máy, chịu ảnh hưởng cache, tải nền và thứ tự A–D; không phải benchmark đa máy hoặc bảo đảm độ trễ triển khai.

## 4.4. Kiểm tra gần trùng và giới hạn holdout

Phân tích độ nhạy dùng TF-IDF ký tự char_wb, ngram=(3,5), max_features=50.000, vocabulary/IDF chỉ fit trên train; tìm láng giềng train gần nhất cho mỗi test bằng cosine. Với từng ngưỡng 0.80/0.90/0.95, bỏ khỏi phần thống kê các test có cosine ≥ ngưỡng, giữ nguyên mô hình và dự đoán, không retune. [Từng cặp ID và cosine](../results/chapter3_4/near_duplicate_pairs.csv) được công bố. Đây là độ giống từ vựng xấp xỉ, chưa xác nhận cùng sự kiện hoặc cùng nguồn; ngưỡng là phân tích thăm dò, không là tiêu chí chọn mô hình.

{sensitivity_table}

{near_note}

Tập kiểm thử là phần chia nội bộ từ split train công khai của ReINTEL. Điểm test đã được quan sát trong báo cáo cũ và nay được sử dụng lại để đối chiếu, vì vậy không gọi đây là một test ngoài hoàn toàn chưa từng nhìn thấy. Nhóm dữ liệu cùng chủ đề/sự kiện có thể tồn tại qua hai phía dù không trùng chính xác. Sự thay đổi điểm khi lọc gần trùng còn chịu thay đổi kích thước, tỷ lệ lớp và độ khó của tập còn lại; không chứng minh toàn bộ chênh lệch là rò rỉ. Chưa có test độc lập theo thời gian, sự kiện hoặc nguồn đăng.

## 4.5. Phân tích FP/FN thực tế

Các ví dụ được chọn bằng quy tắc xác định trước trên cấu hình CV {selected}: 2 FP và 2 FN có |decision_function| nhỏ nhất, cùng 2 FP và 2 FN có |score| lớn nhất trong từng loại lỗi. “Xa” là xa tương đối trong lỗi của chính mô hình đó, không phải xác suất hoặc độ tin cậy đã hiệu chuẩn; không so trực tiếp độ lớn điểm của kernel khác nhau. Nội dung đầy đủ trong [CSV ví dụ lỗi](../results/chapter3_4/error_examples.csv); nhận xét thủ công riêng trong [CSV chú giải](../results/chapter3_4/error_annotations.csv). Nhận xét dựa trên câu chữ, không xác minh tính đúng sai của bài đăng và không kết luận nhãn gốc sai.

{chr(10).join(error_sections)}

## Tệp cho bản in, slide và tái lập

Bản in: metric_comparison (mốc cũ), abcd_metric_comparison, abcd_pairwise_comparison và các confusion_overview. Mỗi hình có PNG nền trắng 300 DPI và SVG nền trắng với chữ chuyển thành đường nét để giữ dấu tiếng Việt trên máy khác; cùng font DejaVu Sans. Kích thước trang xuất được ghi trong từng .export.json. Không gán chuẩn xuất bản của một tạp chí cụ thể; khi đặt hình 2×2 nên dùng chiều rộng toàn trang và ưu tiên SVG.

![Bản slide 16:9 chỉ tập trung F1 và Recall nhãn 1](figures/abcd_slide_16x9.png)

[SVG slide 16:9](figures/abcd_slide_16x9.svg) · [CSV đầy đủ](../results/chapter3_4/metrics_abcd.csv) · [So sánh cặp](../results/chapter3_4/pairwise_comparisons.csv) · [Phân tích độ nhạy](../results/chapter3_4/near_duplicate_sensitivity.csv).

```powershell
.\\.venv\\Scripts\\python.exe -X utf8 run_experiments.py --n-jobs 2
.\\.venv\\Scripts\\python.exe -X utf8 run_evaluation.py --no-render
.\\.venv\\Scripts\\python.exe -X utf8 finalize_experiment.py
.\\.venv\\Scripts\\python.exe -X utf8 run_visuals.py
.\\.venv\\Scripts\\python.exe -X utf8 verify_delivery.py --staging tmp/visuals_review
# Mở và kiểm tra PNG/SVG trong tmp/visuals_review trước khi xuất bản.
.\\.venv\\Scripts\\python.exe -X utf8 run_visuals.py --publish
.\\.venv\\Scripts\\python.exe -X utf8 verify_delivery.py
.\\.venv\\Scripts\\python.exe -X utf8 -m unittest discover -s tests -v
```

## Công cụ trực quan hóa và tài liệu tham khảo

Đã dùng scientific-visualization của K-Dense để thiết kế, kiểm tra thang đo, màu dự phòng bằng họa tiết, xuất bản in/slide và kiểm tra metadata; tiếp tục dùng Matplotlib của dự án. Theo yêu cầu dẫn nguồn của skill: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. [arXiv:2609.00065](https://doi.org/10.48550/arXiv.2609.00065). Metadata tác giả/năm đã đối chiếu bản arXiv hiện hành ngày 14/11/2026; không có journal DOI được ghi ở trang nguồn lúc kiểm tra.
"""
    (docs_dir / "chuong_3_4.md").write_text(content,encoding="utf-8")
    html = """<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Thực nghiệm SVM · Chương 3–4</title><style>
    *{box-sizing:border-box}body{font:16px/1.7 'Segoe UI',Arial,sans-serif;color:#172b4d;margin:0;background:#f3f6fa}main{max-width:1280px;margin:auto;padding:36px;background:white}h1{font-size:34px;line-height:1.25}h2{margin-top:42px;border-top:1px solid #ccd9e5;padding-top:20px}h3{margin-top:30px}a{color:#0072b2}table{border-collapse:collapse;font-size:14px;min-width:100%}th,td{padding:10px;border-bottom:1px solid #dae1e9;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{background:#edf4f9}.scroll{overflow:auto}figure{margin:25px 0}img{max-width:100%;display:block}figcaption{font-size:13px;color:#52647c}code{background:#edf3f8;padding:2px 4px}pre{padding:16px;overflow:auto;background:#edf3f8}p{overflow-wrap:anywhere}@media print{main{padding:0}body{background:white}figure{break-inside:avoid}h2{break-after:avoid}.scroll{overflow:visible}table{white-space:normal;font-size:9px}th,td{white-space:normal;padding:4px}}
    </style><main>""" + markdown_html(content) + "</main></html>"
    (docs_dir / "chapter3_4.html").write_text(html,encoding="utf-8")
