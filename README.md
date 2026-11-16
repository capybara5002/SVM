# Nghiên cứu thuật toán Support Vector Machine giải bài toán nhận diện tin giả trên mạng xã hội

Thực nghiệm Chương 3–4 đã bổ sung thiết kế 2×2 A–D (không/có Underthesea × Linear/RBF), giữ nguyên 3.428 ID train, 856 ID holdout và 5 fold. Xem [báo cáo A–D](docs/chapter3_4.html), [nội dung để đưa vào Chương 3–4](docs/chuong_3_4.md), [bảng kết quả](results/chapter3_4/metrics_abcd.csv), [hình slide 16:9](docs/figures/abcd_slide_16x9.svg) và [biên bản bàn giao](docs/BAN_GIAO.md). Các cấu hình cơ sở/tối ưu có Underthesea được giữ riêng tại [mốc đối chiếu](docs/index.html).

Chạy bổ sung bằng `run_experiments.py --n-jobs 2`; `run_evaluation.py --no-render` đánh giá mốc cũ; `finalize_experiment.py` xuất thống kê và hồ sơ; `run_visuals.py` tạo ảnh tạm; `verify_delivery.py --staging tmp/visuals_review` kiểm tra số liệu/ảnh/liên kết. Sau khi mở và kiểm tra PNG/SVG, dùng `run_visuals.py --publish` để thay đúng các ảnh đã đạt. Danh sách ảnh thay/thêm nằm trong [figure_changes.json](docs/figure_changes.json).

## 1. Mục tiêu đồ án

Đồ án nghiên cứu cách dùng Support Vector Machine (SVM) để phân loại nội dung tiếng Việt trên mạng xã hội thành **thông tin đáng tin cậy** và **thông tin không đáng tin cậy**. Phiên bản hiện tại đã hoàn thành Tuần 1-6: chuẩn bị dữ liệu, huấn luyện SVM Linear/RBF, chọn tham số bằng Grid Search với 5-fold Cross Validation và đánh giá trên tập test đã kiểm tra trùng lặp. Bảng, biểu đồ và dự đoán chi tiết được trình bày tại [docs/index.html](docs/index.html) và [docs/README.md](docs/README.md).

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
│   ├── training_report.py
│   ├── evaluate_svm.py
│   └── evaluation_visuals.py
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
├── docs/
│   ├── index.html
│   ├── README.md
│   ├── evaluation_summary.json
│   ├── figures/
│   ├── tables/
│   └── predictions/
├── tests/
├── download_reintel.py
├── run_preparation.py
├── run_training.py
├── run_evaluation.py
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

## 8. Chạy đánh giá Tuần 6

```powershell
.\.venv\Scripts\python.exe run_evaluation.py
```

Script nạp bốn cấu hình đã lưu: Linear/RBF cơ sở và Linear/RBF tối ưu từ Tuần 5. `svm_best_model.pkl` là bản sao của cấu hình đã chọn từ CV, được kiểm tra trước khi xem kết quả test. Trong quá trình đánh giá không gọi `fit`, không học lại TF-IDF hoặc chọn lại tham số/ngưỡng bằng điểm test.

Trước khi dự đoán, script xác minh hash train khớp metadata huấn luyện, kiểm tra train/test không trùng `source_index`, loại các văn bản test đã xuất hiện trong train và giữ lần đầu nếu văn bản test trùng nội bộ. Quy tắc lọc được thực hiện trước dự đoán; không sửa hai CSV gốc. Toàn bộ mẫu bị loại và nguồn train tương ứng được ghi tại `docs/tables/test_data_audit.csv`.

### Xem bảng và biểu đồ trong docs

- Mở [docs/index.html](docs/index.html) bằng trình duyệt để xem trang tổng hợp chạy ngoại tuyến, bảng chỉ số và biểu đồ theo từng cấu hình.
- [docs/README.md](docs/README.md) là báo cáo Markdown có các hình được nhúng trực tiếp.
- `docs/figures/`: ma trận nhầm lẫn theo số lượng và tỷ lệ, biểu đồ so sánh chỉ số, phân loại lỗi, phân phối dự đoán và so sánh F1 CV/test; mỗi hình có PNG và SVG để chèn vào báo cáo.
- `docs/tables/model_metrics.csv`: Accuracy, Precision/Recall/F1 nhãn 1, F1 macro/weighted, balanced accuracy và số TN/FP/FN/TP.
- `docs/tables/classification_report.csv`: Precision, Recall, F1 và support cho từng nhãn cùng các trung bình macro/weighted.
- `docs/tables/confusion_matrices.csv`: từng ô ma trận nhầm lẫn và tỷ lệ theo hàng.
- `docs/predictions/`: dự đoán từng mẫu và các mẫu sai của từng cấu hình; `all_predictions.csv` tổng hợp cả bốn cấu hình.
- `docs/evaluation_summary.json`: kết quả đầy đủ, cấu hình cố định, phiên bản thư viện và hash đầu vào/mô hình.

Nhãn dương là **1 = không đáng tin cậy**. Ma trận nhầm lẫn có hàng là nhãn thật, cột là nhãn dự đoán, theo thứ tự `[0, 1]`: `[[TN, FP], [FN, TP]]`. Điểm `decision_score` là điểm quyết định của SVC, không phải xác suất. Các bảng test là kết quả trên cùng một tập kiểm thử đã lọc; F1 CV là trung bình năm folds dùng ở Tuần 5 để chọn cấu hình.

### Kết quả test đã hoàn thành

Test gốc có 858 mẫu; loại hai mẫu trùng văn bản với train (source 3768 ↔ 1112, 1807 ↔ 74), còn **856 mẫu**, gồm 711 nhãn 0 và 145 nhãn 1. Test không có văn bản trùng nội bộ. Đã xuất 3.424 dòng dự đoán cho bốn cấu hình và 14 biểu đồ, mỗi biểu đồ có cả PNG và SVG.

| Cấu hình | Accuracy | Precision nhãn 1 | Recall nhãn 1 | F1 nhãn 1 | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Linear cơ sở | 87.50% | 78.79% | 35.86% | 49.29% | 697 | 14 | 93 | 52 |
| RBF cơ sở | 86.10% | 88.24% | 20.69% | 33.52% | 707 | 4 | 115 | 30 |
| Linear tối ưu | 85.63% | 56.18% | 68.97% | 61.92% | 633 | 78 | 45 | 100 |
| RBF tối ưu (đã chọn từ CV) | 83.76% | 51.49% | 71.72% | 59.94% | 613 | 98 | 41 | 104 |

Linear tối ưu có F1 test cao hơn RBF tối ưu trong lần đánh giá này; RBF tối ưu có Recall nhãn 1 cao hơn. Mô hình chính vẫn là RBF đã chọn từ CV ở Tuần 5: không lựa chọn lại mô hình bằng điểm test. Train có một nhóm văn bản giống nhau sau tiền xử lý nhưng nhãn khác nhau; các dòng đó được ghi trong `docs/tables/train_label_conflicts.csv` để nêu giới hạn chất lượng dữ liệu trong báo cáo.

## 9. Kiểm thử

```bat
python -m unittest discover -s tests -v
python -m compileall src download_reintel.py run_preparation.py run_training.py run_evaluation.py tests
```

Dữ liệu tổng hợp nhỏ chỉ xuất hiện trong unit test, không được trình bày như kết quả ReINTEL. Các test kiểm tra tiền xử lý, cấu hình cột, chia 80/20 có phân tầng và từ chỉ xuất hiện ở test không lọt vào từ vựng TF-IDF. Các test huấn luyện kiểm tra từ riêng của validation không vào từ vựng từng fold, văn bản trùng nằm cùng fold, khả năng tái lập, chọn cấu hình theo F1 nhãn 1, dữ liệu lỗi bị từ chối và dự đoán sau lưu/nạp mô hình.

Các test đánh giá xác nhận thứ tự TN/FP/FN/TP, chỉ số nhãn 1, xử lý không có dự đoán dương, loại trùng có thể tái lập, bảo toàn dữ liệu/mô hình gốc và từ chối metadata không khớp. Integration chặn mọi thao tác `fit`/`fit_transform` khi đánh giá và kiểm tra mô hình được chọn từ CV vẫn được giữ nếu một cấu hình khác có F1 test cao hơn. **Toàn bộ 30 kiểm thử đã vượt qua.**

## 10. Web kiểm thử tương tác

Web cho phép **dán nội dung** hoặc **tải tệp TXT, MD, DOCX, PDF có văn bản** và theo dõi toàn bộ pipeline nhận diện. Chạy trong thư mục gốc dự án:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-web.txt
.\.venv\Scripts\python.exe run_web.py
```

Nếu môi trường được tạo bằng `uv` và không có `pip`, cài bằng `uv pip install --python .venv/Scripts/python.exe -r requirements-web.txt`.

Mở **http://127.0.0.1:5000**. Có thể dùng cổng khác: `python run_web.py --port 5050`. Giữ cửa sổ chạy server mở khi dùng web; `Ctrl+C` để dừng. Đây là ứng dụng local, chỉ lắng nghe trên máy cá nhân, chưa phải dịch vụ được triển khai lên Internet.

Sau khi nhấn **Phân tích nội dung**, giao diện nhận cập nhật từ server theo từng bước thực sự hoàn thành:

1. Đọc nội dung gốc, thống kê ký tự và thông tin tệp.
2. Làm sạch: xem văn bản sau từng thao tác Unicode NFC, HTML entity, thẻ HTML, URL, chữ thường và khoảng trắng.
3. Tách từ bằng Underthesea cho C/D; cấu hình A/B giữ văn bản đã làm sạch, đúng như huấn luyện.
4. Transform bằng TF-IDF đã học: xem toàn bộ đặc trưng khác 0, TF, IDF, trọng số chuẩn hóa L2 và từ ngoài từ vựng. Bảng có tìm kiếm và phân trang.
5. Dự đoán nhãn và điểm SVM. Linear hiển thị đóng góp từng đặc trưng; RBF hiển thị tối đa 20 vector hỗ trợ có trị tuyệt đối đóng góp lớn nhất và tổng trên tất cả vector hỗ trợ. Tổng đóng góp cộng hệ số chặn tái tạo điểm quyết định.

Có thể đổi giữa các mô hình **A–D**, mặc định chọn cấu hình thắng theo CV trong `results/chapter3_4/experiment_summary.json` (hiện là D). Tệp mô hình được kiểm tra SHA-256 theo metadata trước khi nạp. Nếu không có artifacts A–D, ứng dụng dùng `artifacts/svm_best_model.pkl` của Tuần 5 khi có. Không fit lại TF-IDF hay SVM trên nội dung nhập, không dùng `artifacts/tfidf_vectorizer.joblib` riêng để ghép vào mô hình khác.

Nhãn giữ nguyên ý nghĩa ReINTEL: **0 = thông tin đáng tin cậy**, **1 = thông tin không đáng tin cậy**. Điểm quyết định **không phải xác suất**. Dự đoán từ mẫu ngôn ngữ không thay thế kiểm chứng sự thật, nhất là với bài báo dài hoặc nội dung khác miền dữ liệu mạng xã hội đã học.

Giới hạn: **5 MB/tệp**, **100.000 ký tự**, **80 trang PDF**. TXT/MD hỗ trợ UTF-8 và UTF-16 có BOM. PDF scan cần OCR bên ngoài; PDF có mật khẩu và tệp hỏng được báo lỗi. Mỗi tệp được xem như một bài viết, không tự tách nhiều bài. Không lưu tệp hay nội dung vào cơ sở dữ liệu/lịch sử; nút **Tải báo cáo JSON** cho phép người dùng chủ động tải toàn bộ nội dung và kết quả về máy. Khi sửa nội dung, tệp hoặc mô hình, kết quả cũ được xóa để tránh nhầm lẫn.

Kiểm thử web, định dạng tệp và sự nhất quán với mô hình:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_web -v
```

## 11. Trạng thái kế hoạch

Tuần 1-3 đã hoàn thành tải dữ liệu thật, EDA, tiền xử lý, tách từ, chia dữ liệu và TF-IDF. Báo cáo Tuần 3 là tài liệu lịch sử của giai đoạn chuẩn bị dữ liệu.

Tuần 4-5 đã hoàn thành huấn luyện cơ sở, so sánh Linear/RBF, chọn `C`, `gamma`, `class_weight`, Grid Search và 5-fold Cross Validation. Đã lưu 5 tệp mô hình `.pkl`, bảng kết quả và [báo cáo Tuần 5](results/bao_cao_tien_do_tuan_5.md). Toàn bộ 17 unit test vượt qua; kiểm tra đầu ra xác nhận 200 lượt fit CV được ghi nhận, các điểm số hữu hạn, mô hình đã refit trên đủ 3.428 mẫu train và nạp lại thành công.

Tuần 6 đã hoàn thành đánh giá bốn cấu hình cố định trên 856 mẫu test sau khi loại hai văn bản trùng train, xuất dự đoán và tính Confusion Matrix, Accuracy, Precision, Recall, F1 cùng các thống kê theo nhãn. [Báo cáo Tuần 6](docs/README.md) và trang HTML trong `docs` hiển thị bảng, sơ đồ quy trình và biểu đồ cho từng cấu hình. Mô hình và hai CSV gốc được bảo toàn; kết quả test không được dùng để lựa chọn lại tham số. Tuần 7-8 sẽ tổng hợp báo cáo đồ án hoàn chỉnh.
