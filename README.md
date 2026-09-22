# Nghiên cứu thuật toán Support Vector Machine giải bài toán nhận diện tin giả trên mạng xã hội

## 1. Mục tiêu đồ án

Đồ án nghiên cứu cách dùng Support Vector Machine (SVM) để phân loại nội dung tiếng Việt trên mạng xã hội thành **thông tin đáng tin cậy** và **thông tin không đáng tin cậy**. Phiên bản hiện tại hoàn thành phạm vi Tuần 1-3: tải và khảo sát dữ liệu thật, làm sạch văn bản, tách từ tiếng Việt, chia dữ liệu và tạo đặc trưng TF-IDF. Dự án **chưa huấn luyện hoặc đánh giá mô hình SVM** và chưa tuyên bố hệ thống có khả năng nhận diện tin giả thành công.

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
│   └── vectorize.py
├── artifacts/
│   ├── tfidf_vectorizer.joblib
│   └── week3_prepared_data.joblib
├── results/
│   ├── eda_summary.txt
│   ├── label_distribution.png
│   ├── preprocessing_examples.txt
│   ├── split_summary.txt
│   └── bao_cao_tien_do_tuan_3.md
├── tests/
├── download_reintel.py
├── run_preparation.py
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

`data/processed/train.csv` và `test.csv` chứa `source_index`, `processed_text`, `label`. `artifacts/week3_prepared_data.joblib` chứa ma trận thưa, nhãn, chỉ số nguồn và tên cột để Tuần 4 không phải lặp lại tiền xử lý.

## 7. Kiểm thử

```bat
python -m unittest discover -s tests -v
python -m compileall src download_reintel.py run_preparation.py tests
```

Dữ liệu tổng hợp nhỏ chỉ xuất hiện trong unit test, không được trình bày như kết quả ReINTEL. Các test kiểm tra xóa URL, khoảng trắng, null, Unicode tiếng Việt, cấu hình cột, chia 80/20 có phân tầng và bảo đảm từ chỉ xuất hiện ở test không lọt vào từ vựng TF-IDF.

## 8. Trạng thái kế hoạch

Đến cuối Tuần 3 đã hoàn thành tải dữ liệu thật, EDA, tiền xử lý, tách từ, chia dữ liệu và TF-IDF chống rò rỉ. Các đầu ra đã sẵn sàng để bắt đầu Tuần 4.

Tuần 4-5 mới thực hiện huấn luyện SVM cơ sở, sau đó lần lượt nghiên cứu Linear/RBF, `C`, `gamma`, Grid Search và 5-Fold Cross Validation. Confusion Matrix, Accuracy, Precision, Recall, F1 và ROC-AUC cuối cùng chưa được thực hiện trong giai đoạn này.

