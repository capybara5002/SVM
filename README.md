# Nghiên cứu thuật toán Support Vector Machine giải bài toán nhận diện tin giả trên mạng xã hội

## 1. Mục tiêu đồ án

Đồ án nghiên cứu cách dùng Support Vector Machine (SVM) để phân loại nội dung tiếng Việt trên mạng xã hội thành **thông tin đáng tin cậy** và **thông tin không đáng tin cậy**. Phiên bản hiện tại đã hoàn thành Tuần 1-5: chuẩn bị dữ liệu, huấn luyện SVM Linear/RBF và chọn tham số bằng Grid Search với 5-fold Cross Validation. Kết quả CV phục vụ chọn mô hình; đánh giá trên tập test độc lập thuộc Tuần 6.

## 2. Bộ dữ liệu ReINTEL

ReINTEL là bộ dữ liệu đa phương thức của shared task VLSP 2020 về nhận diện thông tin có trách nhiệm trên mạng xã hội Việt Nam. Mỗi bài đăng gốc có nội dung chữ, ảnh và một số metadata mạng xã hội. Đồ án này chỉ nghiên cứu văn bản nên không dùng ảnh, lượt thích, bình luận, chia sẻ, thông tin người dùng hoặc đồ thị xã hội.

Nguồn thực tế: [ReliableAI/ReINTEL trên Hugging Face](https://huggingface.co/datasets/ReliableAI/ReINTEL), revision đã tải `5ccbbe27f55f7b848bdd74f012d84e1116dcd657`. Chỉ split `train` được dùng vì đây là split có nhãn; split `test` chính thức không có nhãn và không được dùng làm tập test 20% của đồ án.

Dataset card chính thức xác nhận:

- `post_message`: nội dung văn bản gốc.
- `label`: nhãn được chú thích thủ công.
- `0 = reliable`: thông tin đáng tin cậy.
- `1 = unreliable`: thông tin không đáng tin cậy.

Không có phép đổi giá trị nhãn. Khi xuất CSV dùng trong dự án, `post_message` được đổi tên thành `text`, còn `label` được giữ nguyên.

### Cấu trúc và thống kê dữ liệu thật

Split train tải từ Hugging Face có 4.372 mẫu và 20 cột gốc:

```text
id, image_0, post_message, user_name, timestamp_post,
num_like_post, num_comment_post, num_share_post, label,
image_1, ..., image_11
```

CSV `data/raw/reintel_train.csv` chỉ giữ `text` và `label`. Kết quả được tính tự động bởi pipeline:

- Nhãn 0: 3.638 mẫu (83,21%).
- Nhãn 1: 734 mẫu (16,79%).
- Thiếu văn bản: 1; thiếu nhãn: 0.
- Dòng trùng hoàn toàn: 81.
- Độ dài văn bản: nhỏ nhất 5, lớn nhất 28.453, trung bình 747,95 và trung vị 233 ký tự.

Chi tiết có thể kiểm chứng tại `results/eda_summary.txt`. Metadata về nguồn, revision, lược đồ gốc và ánh xạ nhãn nằm trong `data/raw/reintel_metadata.json`.

## 3. Cấu trúc dự án

```text
SVM/
├── data/
│   ├── raw/
│   │   ├── reintel_train.csv
│   │   └── reintel_metadata.json
│   └── processed/
│       ├── train.csv
│       └── test.csv
├── notebooks/
├── src/
│   ├── config.py
│   ├── load_data.py
│   ├── eda.py
│   ├── preprocess.py
│   ├── split_data.py
│   ├── vectorize.py
│   ├── train_svm.py
│   └── training_report.py
├── artifacts/
│   ├── tfidf_vectorizer.joblib
│   ├── week3_prepared_data.joblib
│   ├── svm_linear_baseline.pkl
│   ├── svm_rbf_baseline.pkl
│   ├── svm_linear_best.pkl
│   ├── svm_rbf_best.pkl
│   └── svm_best_model.pkl
├── results/
│   ├── eda_summary.txt
│   ├── label_distribution.png
│   ├── preprocessing_examples.txt
│   ├── split_summary.txt
│   ├── bao_cao_tien_do_tuan_3.md
│   ├── bao_cao_tien_do_tuan_5.md
│   └── week4_5/
│       ├── model_comparison.csv
│       ├── grid_search_linear.csv
│       ├── grid_search_rbf.csv
│       ├── baseline_cv_folds.csv
│       ├── best_cv_folds.csv
│       ├── cv_fold_assignments.csv
│       └── training_summary.json
├── tests/
├── download_reintel.py
├── run_preparation.py
├── run_training.py
└── requirements.txt
```

## 4. Môi trường Python

Khuyến nghị Python 3.11. Trên Windows:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Trong PowerShell có thể kích hoạt bằng `.venv\Scripts\Activate.ps1`. Nếu máy chưa có lệnh `python` nhưng đã có `uv`:

```bat
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

Kiểm tra thư viện:

```bat
python -c "import pandas, numpy, sklearn, underthesea, datasets; print('Imports OK')"
```

## 5. Tải lại ReINTEL

```bat
python download_reintel.py
```

Script thực hiện các kiểm soát sau:

1. Lấy revision hiện hành từ Hugging Face và khóa revision đó khi tải.
2. Đọc dataset card chính thức, chỉ tiếp tục nếu xác minh được `0 = reliable`, `1 = unreliable`.
3. In features, 20 tên cột gốc, số bản ghi và 5 bản ghi rút gọn.
4. Chỉ trích xuất `post_message` và `label`, sau đó lưu thành `text`, `label` với UTF-8.
5. Lưu metadata nguồn để EDA có thể báo cáo cả lược đồ gốc.

Nếu tải lỗi, script in nguyên loại lỗi và thông báo, không tạo dữ liệu giả. Có thể tải thủ công tệp train Parquet tại trang **Files and versions** của nguồn trên, chỉ lấy hai cột `post_message`, `label`, đổi `post_message` thành `text`, rồi lưu CSV UTF-8 tại `data/raw/reintel_train.csv`. File phải có đúng hai cột `text` và `label`; không tự suy đoán hay đổi ý nghĩa nhãn.

## 6. Chạy pipeline Tuần 1-3

```bat
python run_preparation.py
```

Quy trình:

```text
ReINTEL thật
→ Làm sạch văn bản
→ Tách từ bằng Underthesea
→ Chia train/test 80/20 có phân tầng
→ Fit TF-IDF trên train
→ Transform train và test
→ Sẵn sàng cho SVM
```

### Làm sạch và tách từ

`clean_text` chuẩn hóa Unicode NFC, chuyển chữ thường, bỏ URL/thẻ HTML và gom khoảng trắng. Quy tắc bảo thủ nên vẫn giữ dấu tiếng Việt, từ phủ định như “không”, “chưa”, “chẳng”, dấu câu và emoji. Sau làm sạch, `underthesea.word_tokenize(text, format="text")` được dùng để tách từ tiếng Việt. Mười ví dụ thật gốc → làm sạch → tách từ nằm trong `results/preprocessing_examples.txt`.

### Tại sao phải chia train/test?

Tập train dùng để học biểu diễn và mô hình; tập test được giữ độc lập để đánh giá khả năng tổng quát hóa trên dữ liệu chưa thấy. Pipeline dùng `random_state=42` để tái lập và `stratify=label` để tỷ lệ hai nhãn trong mỗi tập gần với dữ liệu ban đầu.

Sau khi loại dữ liệu thiếu, trùng và văn bản rỗng sau làm sạch, còn 4.286 mẫu:

- Train: 3.428 mẫu, gồm 2.851 nhãn 0 và 577 nhãn 1.
- Test: 858 mẫu, gồm 713 nhãn 0 và 145 nhãn 1.

### TF-IDF và rò rỉ dữ liệu

Pipeline chia dữ liệu **trước** khi tạo từ vựng TF-IDF:

```python
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
```

TF-IDF tuyệt đối không được fit trên toàn bộ dữ liệu hoặc fit lại trên test. Nếu từ vựng/trọng số IDF học từ test, thông tin của tập đánh giá đã đi vào quá trình chuẩn bị mô hình; đó là **rò rỉ dữ liệu (data leakage)** và làm kết quả đánh giá sau này không đáng tin cậy.

Kết quả hiện tại:

- `X_train_tfidf.shape = (3428, 10000)`.
- `X_test_tfidf.shape = (858, 10000)`.
- Từ vựng có 10.000 đặc trưng.
- Hai ma trận là ma trận thưa CSR.
- `artifacts/tfidf_vectorizer.joblib` chỉ được fit trên train.

`data/processed/train.csv` và `test.csv` chứa `source_index`, `processed_text`, `label`. `artifacts/week3_prepared_data.joblib` chứa ma trận thưa, nhãn, chỉ số nguồn và tên cột. Tuần 4-5 tái sử dụng văn bản đã tiền xử lý trong `train.csv`, nhưng học lại TF-IDF trong từng fold để phần validation không tham gia tạo từ vựng/IDF.

## 7. Chạy huấn luyện Tuần 4-5

Trong PowerShell, dùng trực tiếp môi trường ảo:

```powershell
.\.venv\Scripts\python.exe run_training.py --n-jobs 4
```

Hoặc sau khi kích hoạt môi trường: `python run_training.py`. Mặc định chạy 2 tác vụ song song; thêm `--verbose 2` để xem tiến độ từng lượt fit. Script chỉ cần `data/processed/train.csv`, không đọc `test.csv` hoặc ma trận TF-IDF đã fit sẵn.

Quy trình thực nghiệm:

1. Kiểm tra văn bản, nhãn 0/1, chỉ số nguồn và số mẫu đủ cho 5-fold CV.
2. Tạo fold có phân tầng với `random_state=42`. Khi có văn bản trùng, dùng `StratifiedGroupKFold` để các văn bản giống nhau nằm cùng fold; nếu không trùng, dùng `StratifiedKFold`.
3. Chạy mô hình cơ sở: Dummy lớp đa số, `SVC(kernel="linear", C=1)` và `SVC(kernel="rbf", C=1, gamma="scale")`.
4. Grid Search Linear/RBF trên cùng các folds, với pipeline `TF-IDF → SVC` được fit lại trong từng fold.
5. Chọn theo **F1 của nhãn 1 (unreliable)** vì dữ liệu mất cân bằng. Lưu thêm Precision/Recall nhãn 1, Accuracy, F1 macro và balanced accuracy.
6. Refit cấu hình tốt nhất của mỗi kernel trên toàn bộ train, lưu mô hình và kiểm tra nạp lại. Điểm CV là điểm dùng để chọn tham số; chưa phải kết quả test độc lập.

| Tham số | Linear | RBF |
|---|---|---|
| C | 0.1, 1, 10, 100 | 0.1, 1, 10, 100 |
| gamma | Không áp dụng | scale, 0.01, 0.1, 1 |
| class_weight | None, balanced | None, balanced |
| Số cấu hình | 8 | 32 |
| Số lượt fit CV | 40 | 160 |

Có thêm 15 lượt fit CV cho 3 mô hình cơ sở và các lượt refit trên toàn bộ train. Cấu hình nằm trong `src/config.py`.

Toàn bộ kết quả CV, điểm từng fold, thời gian fit, tham số được chọn, hash dữ liệu và phiên bản thư viện được lưu tại `results/week4_5/`. Báo cáo sinh từ kết quả thật tại `results/bao_cao_tien_do_tuan_5.md`.

### Kết quả lần chạy đã hoàn thành

Thực nghiệm trên 3.428 mẫu train, 5 folds, 40 cấu hình Grid Search, tổng cộng 200 lượt fit CV cho Grid Search. F1, Precision và Recall bên dưới tính cho nhãn 1 và lấy trung bình qua 5 folds.

| Mô hình | F1 nhãn 1 | Precision nhãn 1 | Recall nhãn 1 | Accuracy |
|---|---:|---:|---:|---:|
| SVM Linear cơ sở | 0.4957 | 0.7697 | 0.3698 | 0.8740 |
| SVM RBF cơ sở | 0.3366 | 0.8656 | 0.2110 | 0.8608 |
| SVM Linear sau Grid Search | 0.5869 | 0.5699 | 0.6096 | 0.8562 |
| SVM RBF sau Grid Search | 0.5934 | 0.5260 | 0.6838 | 0.8433 |

Chọn **RBF, C=10, gamma=0.01, class_weight=balanced** theo F1 nhãn 1 cao nhất trong lưới đã khảo sát. F1 CV có độ lệch chuẩn 0.0538; đây là điểm chọn tham số, chưa phải kết quả trên test. Hai mô hình tối ưu có điểm F1 gần nhau. RBF tối ưu có Recall cao hơn, trong khi Linear tối ưu có Precision và Accuracy cao hơn.

### Nạp mô hình để sử dụng

Tệp `.pkl` được lưu bằng Joblib và chứa cả TF-IDF lẫn SVM. Đầu vào là văn bản đã làm sạch và tách từ giống khi huấn luyện:

```python
import joblib
from src.preprocess import segment_vietnamese

model = joblib.load("artifacts/svm_best_model.pkl")
processed = segment_vietnamese("Nội dung bài đăng cần phân loại")
predicted_label = int(model.predict([processed])[0])
decision_score = float(model.decision_function([processed])[0])
```

`0 = reliable`, `1 = unreliable`. `decision_score` là khoảng cách quyết định, không phải xác suất. Các mô hình này không cần dùng `tfidf_vectorizer.joblib` riêng của Tuần 3.

## 8. Kiểm thử

```bat
python -m unittest discover -s tests -v
python -m compileall src download_reintel.py run_preparation.py run_training.py tests
```

Dữ liệu tổng hợp nhỏ chỉ xuất hiện trong unit test, không được trình bày như kết quả ReINTEL. Các test kiểm tra tiền xử lý, cấu hình cột, chia 80/20 có phân tầng và từ chỉ xuất hiện ở test không lọt vào từ vựng TF-IDF. Các test huấn luyện kiểm tra từ riêng của validation không vào từ vựng từng fold, văn bản trùng nằm cùng fold, khả năng tái lập, chọn cấu hình theo F1 nhãn 1, dữ liệu lỗi bị từ chối và dự đoán sau lưu/nạp mô hình.

## 9. Trạng thái kế hoạch

Tuần 1-3 đã hoàn thành tải dữ liệu thật, EDA, tiền xử lý, tách từ, chia dữ liệu và TF-IDF. Báo cáo Tuần 3 là tài liệu lịch sử của giai đoạn chuẩn bị dữ liệu.

Tuần 4-5 đã hoàn thành huấn luyện cơ sở, so sánh Linear/RBF, chọn `C`, `gamma`, `class_weight`, Grid Search và 5-fold Cross Validation. Đã lưu 5 tệp mô hình `.pkl`, bảng kết quả và [báo cáo Tuần 5](results/bao_cao_tien_do_tuan_5.md). Toàn bộ 17 unit test vượt qua; kiểm tra đầu ra xác nhận 200 lượt fit CV được ghi nhận, các điểm số hữu hạn, mô hình đã refit trên đủ 3.428 mẫu train và nạp lại thành công.

Tuần 6 sẽ đánh giá mô hình cố định trên tập test: Confusion Matrix, Accuracy, Precision, Recall và F1. Kiểm tra ban đầu phát hiện 2 văn bản đã tiền xử lý xuất hiện ở cả train và test; cần xử lý và ghi lại việc này trước khi đánh giá tính tổng quát hóa. Chưa dùng điểm test để lựa chọn mô hình. Tuần 7-8 sẽ tổng hợp báo cáo đồ án hoàn chỉnh.
