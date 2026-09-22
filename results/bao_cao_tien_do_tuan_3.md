# Báo cáo tiến độ đến cuối Tuần 3

## Tên đề tài

**Nghiên cứu thuật toán Support Vector Machine giải bài toán nhận diện tin giả trên mạng xã hội**

## Giai đoạn 1 - Tuần 1

- Tìm hiểu nguyên lý cơ bản của SVM: siêu phẳng phân tách, biên, vector hỗ trợ và khái niệm kernel. Giai đoạn này chưa huấn luyện mô hình.
- Tìm hiểu TF-IDF và cách chuyển nội dung văn bản thành đặc trưng số.
- Tìm hiểu ReINTEL của VLSP 2020 và bài toán phân loại thông tin đáng tin cậy/không đáng tin cậy trên mạng xã hội.
- Chuẩn bị Python 3.11, cấu trúc dự án, VS Code/Jupyter và môi trường ảo `.venv`.
- Cài đặt và kiểm tra pandas, NumPy, scikit-learn, Underthesea, datasets, PyArrow, Matplotlib, Jupyter và Joblib.

## Giai đoạn 2 - Tuần 2-3

- Tải dữ liệu thật từ `ReliableAI/ReINTEL` trên Hugging Face, revision `5ccbbe27f55f7b848bdd74f012d84e1116dcd657`.
- Xác minh split train có 4.372 mẫu và 20 cột gốc. Đồ án chỉ dùng `post_message` và `label`, được lưu nội bộ thành `text` và `label`; không sử dụng ảnh hay metadata mạng xã hội.
- Xác minh trên dataset card: nhãn 0 là thông tin đáng tin cậy (`reliable`), nhãn 1 là thông tin không đáng tin cậy (`unreliable`). Không đổi giá trị nhãn.
- Khảo sát dữ liệu bằng code: 3.638 mẫu nhãn 0 (83,21%) và 734 mẫu nhãn 1 (16,79%); thiếu 1 văn bản, không thiếu nhãn và có 81 dòng trùng hoàn toàn.
- Thống kê độ dài văn bản theo ký tự: nhỏ nhất 5, lớn nhất 28.453, trung bình 747,95 và trung vị 233.
- Làm sạch bằng Unicode NFC và Regex: chữ thường, bỏ URL/thẻ HTML, gom khoảng trắng nhưng giữ dấu tiếng Việt, từ phủ định, dấu câu và emoji.
- Tách từ tiếng Việt bằng `underthesea.word_tokenize(..., format="text")`; lưu 10 ví dụ thật trước/sau xử lý.
- Sau khi loại dòng thiếu, trùng và văn bản rỗng sau làm sạch, còn 4.286 mẫu hợp lệ.
- Chia phân tầng 80/20 với `random_state=42`: train có 3.428 mẫu (2.851 nhãn 0, 577 nhãn 1); test có 858 mẫu (713 nhãn 0, 145 nhãn 1).
- Fit TF-IDF unigram tối đa 10.000 đặc trưng chỉ trên train, sau đó transform test. Ma trận train có kích thước `(3428, 10000)`, test `(858, 10000)`, đều là ma trận thưa CSR.
- Lưu CSV đã xử lý, TF-IDF vectorizer và ma trận/nhãn để sẵn sàng huấn luyện SVM ở Tuần 4.
- Chạy kiểm tra cú pháp, kiểm tra phụ thuộc và 12 unit test; tất cả đều thành công. Kiểm tra mã nguồn cũng xác nhận không có bước huấn luyện/đánh giá SVM trong giai đoạn này.

## Kết quả và giới hạn hiện tại

Các thống kê trên được sinh từ `data/raw/reintel_train.csv` bởi pipeline và có thể kiểm tra tại `results/eda_summary.txt`, `results/split_summary.txt`. Đến thời điểm này nhóm **chỉ hoàn thành chuẩn bị dữ liệu**. Nhóm chưa huấn luyện SVM, chưa có Accuracy, Precision, Recall, F1, ROC-AUC hoặc kết quả nhận diện tin giả và không đưa ra số liệu hiệu năng giả định.

## Nếu giảng viên hỏi "Nhóm đã làm đến đâu?"

Nhóm đã hoàn thành phần việc từ Tuần 1 đến hết Tuần 3. Nhóm đã tải và xác minh bộ ReINTEL thật từ Hugging Face, trong đó split có nhãn gồm 4.372 bài đăng. Nhóm chỉ sử dụng nội dung chữ và nhãn, sau đó làm sạch văn bản và tách từ tiếng Việt bằng Underthesea. Dữ liệu hợp lệ được chia phân tầng 80/20 thành 3.428 mẫu train và 858 mẫu test. TF-IDF chỉ được học trên tập train để tránh rò rỉ dữ liệu, rồi dùng cùng vectorizer để biến đổi tập test. Các tệp đã xử lý và ma trận TF-IDF hiện đã sẵn sàng cho việc huấn luyện mô hình SVM cơ sở ở Tuần 4. Nhóm chưa báo cáo chất lượng mô hình vì chưa thực hiện giai đoạn huấn luyện và đánh giá.

## Kế hoạch Tuần 4-5

Tuần 4 sẽ huấn luyện mô hình SVM cơ sở trước. Sau đó mới thực hiện so sánh Linear/RBF, khảo sát `C`, `gamma`, Grid Search và 5-Fold Cross Validation trên dữ liệu huấn luyện. Các chỉ số đánh giá cuối cùng chỉ được tính ở giai đoạn thích hợp, với tập test được giữ độc lập.
