"""Sinh bao cao tu ket qua huan luyen thuc te, khong dung so lieu test."""

from pathlib import Path


def _comparison_row(name: str, metrics: dict) -> str:
    return (
        f"| {name} | {metrics['f1']['mean']:.4f} ± {metrics['f1']['std']:.4f} | "
        f"{metrics['precision']['mean']:.4f} | {metrics['recall']['mean']:.4f} | "
        f"{metrics['accuracy']['mean']:.4f} | {metrics['f1_macro']['mean']:.4f} |"
    )


def write_training_report(summary: dict, path: Path) -> None:
    best_kernel = summary["best_model"]["kernel"]
    best = summary["searches"][best_kernel]
    rows = [
        _comparison_row(name, result["metrics"])
        for name, result in summary["baselines"].items()
    ]
    rows.extend(
        _comparison_row(f"SVM {kernel} sau Grid Search", result["metrics"])
        for kernel, result in summary["searches"].items()
    )
    parameters, folds = [], []
    for kernel, result in summary["searches"].items():
        params = result["best_params"]
        parameters.append(
            f"| {kernel} | {params['svc__C']} | {params.get('svc__gamma', 'Không áp dụng')} | "
            f"{params['svc__class_weight']} | {result['candidates']} | {result['cv_fits']} |"
        )
    for details in summary["cv"]["details"]:
        folds.append(
            f"| {details['fold']} | {details['fit_rows']} | {details['validation_rows']} | "
            f"{details['validation_labels']['0']} | {details['validation_labels']['1']} |"
        )
    grids = "; ".join(
        f"{kernel}: {result['parameter_grid']}"
        for kernel, result in summary["searches"].items()
    )
    content = f"""# Báo cáo tiến độ đến cuối Tuần 5

Đề tài: **Nghiên cứu thuật toán Support Vector Machine giải bài toán nhận diện tin giả trên mạng xã hội**.

Kết quả được sinh từ lần chạy hoàn tất lúc **{summary['completed_at']}** (UTC+7).

## 1. Công việc đã hoàn thành

- Huấn luyện SVM cơ sở với kernel Linear và RBF trên dữ liệu ReINTEL đã tiền xử lý.
- Thực hiện {summary['cv']['folds']}-fold Cross Validation có phân tầng, seed {summary['cv']['random_state']}.
- Grid Search khảo sát `C`, `gamma` và `class_weight` (None/balanced) để xem ảnh hưởng của mất cân bằng nhãn.
- Lưu mô hình cơ sở, mô hình tối ưu từng kernel và mô hình tốt nhất dưới dạng `.pkl` bằng Joblib.
- Mỗi tệp mô hình chứa cả TF-IDF và SVM; đã nạp lại và kiểm tra dự đoán trên 10 mẫu train.

## 2. Thiết kế thực nghiệm

Chỉ dùng **{summary['train_rows']:,} mẫu train**: {summary['train_labels']['0']:,} nhãn 0 (đáng tin cậy) và {summary['train_labels']['1']:,} nhãn 1 (không đáng tin cậy). Script huấn luyện không đọc hoặc đánh giá `test.csv`.

Pipeline `TF-IDF → SVC` được fit lại trong từng fold. Không đưa ma trận TF-IDF đã fit trên toàn bộ train ở Tuần 3 trực tiếp vào Cross Validation: cách đó khiến từ vựng/IDF của phần validation tham gia huấn luyện.

Train có {summary['duplicate_processed_text_rows']} dòng trùng văn bản sau tiền xử lý. Bộ chia **{summary['cv']['type']}** giữ các văn bản giống nhau trong cùng fold, đồng thời cố gắng giữ tỷ lệ nhãn. Các cấu hình dùng cùng một danh sách fold để có thể so sánh.

| Fold | Mẫu fit | Mẫu validation | Validation nhãn 0 | Validation nhãn 1 |
|---|---:|---:|---:|---:|
{chr(10).join(folds)}

Chọn tham số theo **F1 của nhãn 1**, là lớp thiểu số cần nhận diện. Precision và Recall bên dưới cũng tính cho nhãn 1. Accuracy, F1 macro và balanced accuracy được lưu để bổ sung góc nhìn. Dummy luôn dự đoán lớp đa số là mốc so sánh đơn giản.

Lưới tham số thực tế: `{grids}`.

## 3. Kết quả Cross Validation trên train

Các số là trung bình qua {summary['cv']['folds']} folds; độ lệch chuẩn F1 dùng `ddof=0`.

| Mô hình | F1 nhãn 1 ± std | Precision nhãn 1 | Recall nhãn 1 | Accuracy | F1 macro |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## 4. Tham số được chọn

| Kernel | C | gamma | class_weight | Số cấu hình | Số lượt fit CV |
|---|---:|---|---|---:|---:|
{chr(10).join(parameters)}

Mô hình được chọn là **SVM {best_kernel.upper()}**, F1 CV nhãn 1 **{summary['best_model']['mean_cv_f1']:.4f}**. Pipeline được refit trên toàn bộ train sau khi chọn tham số. TF-IDF cuối cùng có {best['fitted_features']:,} đặc trưng; số support vector theo nhãn 0/1 là {best['support_vectors_by_class']}.

Các tệp `.pkl` nhận **văn bản đã làm sạch và tách từ bằng Underthesea**, sau đó tự transform bằng TF-IDF bên trong pipeline. Chúng không cần vectorizer Tuần 3 riêng.

## 5. Đầu ra và khả năng tái lập

- `artifacts/svm_linear_baseline.pkl`, `artifacts/svm_rbf_baseline.pkl`: mô hình cơ sở.
- `artifacts/svm_linear_best.pkl`, `artifacts/svm_rbf_best.pkl`: mô hình tối ưu theo từng kernel.
- `artifacts/svm_best_model.pkl`: mô hình được chọn cho Tuần 6.
- `results/week4_5/model_comparison.csv`: bảng so sánh kết quả CV.
- `results/week4_5/grid_search_linear.csv`, `grid_search_rbf.csv`: toàn bộ cấu hình, điểm từng fold và thời gian fit.
- `results/week4_5/baseline_cv_folds.csv`, `best_cv_folds.csv`: điểm từng fold của mô hình cơ sở và cấu hình tối ưu.
- `results/week4_5/cv_fold_assignments.csv`: ánh xạ `source_index` sang fold validation.
- `results/week4_5/training_summary.json`: cấu hình, phiên bản thư viện và thông tin mô hình.

SHA-256 của train.csv: `{summary['train_sha256']}`.

Phiên bản: Python {summary['versions']['python']}, scikit-learn {summary['versions']['scikit_learn']}, NumPy {summary['versions']['numpy']}, pandas {summary['versions']['pandas']}, Joblib {summary['versions']['joblib']}.

Thời gian chạy tổng cộng: {summary['total_seconds']:.1f} giây, `n_jobs={summary['n_jobs']}`. Chạy lại bằng `python run_training.py`.

## 6. Giới hạn và kế hoạch Tuần 6

Điểm Grid Search được dùng để chọn tham số nên có thể lạc quan do chọn cấu hình tốt nhất trên cùng các folds; đây **chưa phải chất lượng trên tập test độc lập**. Chưa có Confusion Matrix hoặc số liệu đánh giá cuối cùng của Tuần 6.

Tuần 6 sẽ cố định mô hình đã chọn, kiểm tra tính độc lập của tập test (bao gồm văn bản trùng sau tiền xử lý), rồi đánh giá Accuracy, Precision, Recall, F1 và Confusion Matrix. Không dùng kết quả test để chọn lại tham số. Nhãn ReINTEL thể hiện đáng tin cậy/không đáng tin cậy; kết quả này chưa chứng minh khả năng kiểm chứng sự thật của từng bài đăng.

## Tài liệu phương pháp

- [scikit-learn: tránh rò rỉ dữ liệu và dùng Pipeline trong CV](https://scikit-learn.org/1.7/common_pitfalls.html#data-leakage).
- [GridSearchCV và refit theo scorer được chọn](https://scikit-learn.org/1.7/modules/generated/sklearn.model_selection.GridSearchCV.html).
- [StratifiedGroupKFold: phân tầng với nhóm không giao nhau](https://scikit-learn.org/1.7/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).
- [SVC: C, gamma và class_weight](https://scikit-learn.org/1.7/modules/generated/sklearn.svm.SVC.html).
"""
    path.write_text(content, encoding="utf-8")
