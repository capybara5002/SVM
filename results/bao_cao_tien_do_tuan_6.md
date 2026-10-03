# Báo cáo tiến độ Tuần 6

Hoàn tất đánh giá lúc 2026-10-03T16:25:55+07:00 (UTC+7).

Test gốc 858 mẫu; loại 2 mẫu theo quy tắc kiểm tra trùng, còn 856 mẫu. Đã đánh giá bốn cấu hình SVM cố định, xuất dự đoán, bảng chỉ số, báo cáo theo nhãn và biểu đồ.

Mô hình đã chọn từ CV: **RBF tối ưu**, {'kernel': 'rbf', 'C': 10.0, 'gamma': 0.01, 'class_weight': 'balanced'}.

| Accuracy | Precision nhãn 1 | Recall nhãn 1 | F1 nhãn 1 |
|---:|---:|---:|---:|
| 0.8376 | 0.5149 | 0.7172 | 0.5994 |

[Xem báo cáo đầy đủ, bảng và biểu đồ](../docs/README.md). [Mở trang tổng hợp](../docs/index.html).

Không huấn luyện lại hoặc lựa chọn lại tham số từ kết quả test. Tuần 7–8 sẽ tổng hợp báo cáo đồ án.
