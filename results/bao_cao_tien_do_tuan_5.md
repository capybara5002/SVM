# Báo cáo tiến độ đến cuối Tuần 5

Đề tài: **Nghiên cứu thuật toán Support Vector Machine giải bài toán nhận diện tin giả trên mạng xã hội**.

Kết quả được sinh từ lần chạy hoàn tất lúc **2026-10-03T15:47:51+07:00** (UTC+7).

## 1. Công việc đã hoàn thành

- Huấn luyện SVM cơ sở với kernel Linear và RBF trên dữ liệu ReINTEL đã tiền xử lý.
- Thực hiện 5-fold Cross Validation có phân tầng, seed 42.
- Grid Search khảo sát `C`, `gamma` và `class_weight` (None/balanced) để xem ảnh hưởng của mất cân bằng nhãn.
- Lưu mô hình cơ sở, mô hình tối ưu từng kernel và mô hình tốt nhất dưới dạng `.pkl` bằng Joblib.
- Mỗi tệp mô hình chứa cả TF-IDF và SVM; đã nạp lại và kiểm tra dự đoán trên 10 mẫu train.

## 2. Thiết kế thực nghiệm

Chỉ dùng **3,428 mẫu train**: 2,851 nhãn 0 (đáng tin cậy) và 577 nhãn 1 (không đáng tin cậy). Script huấn luyện không đọc hoặc đánh giá `test.csv`.

Pipeline `TF-IDF → SVC` được fit lại trong từng fold. Không đưa ma trận TF-IDF đã fit trên toàn bộ train ở Tuần 3 trực tiếp vào Cross Validation: cách đó khiến từ vựng/IDF của phần validation tham gia huấn luyện.

Train có 4 dòng trùng văn bản sau tiền xử lý. Bộ chia **StratifiedGroupKFold** giữ các văn bản giống nhau trong cùng fold, đồng thời cố gắng giữ tỷ lệ nhãn. Các cấu hình dùng cùng một danh sách fold để có thể so sánh.

| Fold | Mẫu fit | Mẫu validation | Validation nhãn 0 | Validation nhãn 1 |
|---|---:|---:|---:|---:|
| 1 | 2741 | 687 | 578 | 109 |
| 2 | 2744 | 684 | 561 | 123 |
| 3 | 2743 | 685 | 584 | 101 |
| 4 | 2743 | 685 | 553 | 132 |
| 5 | 2741 | 687 | 575 | 112 |

Chọn tham số theo **F1 của nhãn 1**, là lớp thiểu số cần nhận diện. Precision và Recall bên dưới cũng tính cho nhãn 1. Accuracy, F1 macro và balanced accuracy được lưu để bổ sung góc nhìn. Dummy luôn dự đoán lớp đa số là mốc so sánh đơn giản.

Lưới tham số thực tế: `linear: {'svc__C': [0.1, 1.0, 10.0, 100.0], 'svc__class_weight': [None, 'balanced']}; rbf: {'svc__C': [0.1, 1.0, 10.0, 100.0], 'svc__class_weight': [None, 'balanced'], 'svc__gamma': ['scale', 0.01, 0.1, 1.0]}`.

## 3. Kết quả Cross Validation trên train

Các số là trung bình qua 5 folds; độ lệch chuẩn F1 dùng `ddof=0`.

| Mô hình | F1 nhãn 1 ± std | Precision nhãn 1 | Recall nhãn 1 | Accuracy | F1 macro |
|---|---:|---:|---:|---:|---:|
| dummy_majority | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.8317 | 0.4540 |
| svm_linear_baseline | 0.4957 ± 0.0172 | 0.7697 | 0.3698 | 0.8740 | 0.7118 |
| svm_rbf_baseline | 0.3366 ± 0.0420 | 0.8656 | 0.2110 | 0.8608 | 0.6294 |
| SVM linear sau Grid Search | 0.5869 ± 0.0454 | 0.5699 | 0.6096 | 0.8562 | 0.7499 |
| SVM rbf sau Grid Search | 0.5934 ± 0.0538 | 0.5260 | 0.6838 | 0.8433 | 0.7481 |

## 4. Tham số được chọn

| Kernel | C | gamma | class_weight | Số cấu hình | Số lượt fit CV |
|---|---:|---|---|---:|---:|
| linear | 1.0 | Không áp dụng | balanced | 8 | 40 |
| rbf | 10.0 | 0.01 | balanced | 32 | 160 |

Mô hình được chọn là **SVM RBF**, F1 CV nhãn 1 **0.5934**. Pipeline được refit trên toàn bộ train sau khi chọn tham số. TF-IDF cuối cùng có 10,000 đặc trưng; số support vector theo nhãn 0/1 là [2112, 469].

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

SHA-256 của train.csv: `9b429b3d7697bb89b71760968d3324b32e01892ef1b140009c1e0f544da17204`.

Phiên bản: Python 3.11.16, scikit-learn 1.7.2, NumPy 2.3.3, pandas 2.3.2, Joblib 1.5.2.

Thời gian chạy tổng cộng: 276.4 giây, `n_jobs=4`. Chạy lại bằng `python run_training.py`.

## 6. Giới hạn và kế hoạch Tuần 6

Điểm Grid Search được dùng để chọn tham số nên có thể lạc quan do chọn cấu hình tốt nhất trên cùng các folds; đây **chưa phải chất lượng trên tập test độc lập**. Chưa có Confusion Matrix hoặc số liệu đánh giá cuối cùng của Tuần 6.

Tuần 6 sẽ cố định mô hình đã chọn, kiểm tra tính độc lập của tập test (bao gồm văn bản trùng sau tiền xử lý), rồi đánh giá Accuracy, Precision, Recall, F1 và Confusion Matrix. Không dùng kết quả test để chọn lại tham số. Nhãn ReINTEL thể hiện đáng tin cậy/không đáng tin cậy; kết quả này chưa chứng minh khả năng kiểm chứng sự thật của từng bài đăng.

## Tài liệu phương pháp

- [scikit-learn: tránh rò rỉ dữ liệu và dùng Pipeline trong CV](https://scikit-learn.org/1.7/common_pitfalls.html#data-leakage).
- [GridSearchCV và refit theo scorer được chọn](https://scikit-learn.org/1.7/modules/generated/sklearn.model_selection.GridSearchCV.html).
- [StratifiedGroupKFold: phân tầng với nhóm không giao nhau](https://scikit-learn.org/1.7/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).
- [SVC: C, gamma và class_weight](https://scikit-learn.org/1.7/modules/generated/sklearn.svm.SVC.html).
