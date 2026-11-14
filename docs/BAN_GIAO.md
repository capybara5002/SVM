# Bàn giao thực nghiệm SVM Chương 3–4

Hoàn tất thực nghiệm ngày 14/11/2026. [Báo cáo HTML A–D](chapter3_4.html), [nội dung Chương 3–4](chuong_3_4.md), [mốc cũ](index.html), [CSV kết quả](../results/chapter3_4/metrics_abcd.csv).

## Kết quả đủ căn cứ sử dụng

- Đã chạy đủ 400 lượt fit CV và 4 refit A–D trên cùng 3.428 ID train, 856 ID holdout và 5 fold. ID/nhãn khớp dữ liệu nguồn; văn bản sau cleaning không trùng chính xác qua ranh giới train/test hoặc train/validation của từng fold. [Biên bản dữ liệu](../results/chapter3_4/data_audit.json).
- TF-IDF nằm trong pipeline và fit riêng trên train mỗi fold. A/B dùng regex Unicode `(?u)\b\w+\b`, không phải split khoảng trắng thuần túy. C/D dùng thêm Underthesea. Tham số được chọn theo F1 nhãn 1 trung bình CV; toàn bộ lựa chọn được ghi trước bước chấm holdout.
- D được chọn theo CV: F1 = 0,5934 ± 0,0538 (SD của 5 fold, ddof=0). F1 test của A/B/C/D lần lượt là 0,5383 / 0,5401 / 0,6192 / 0,5994. C có F1 test cao nhất về mặt mô tả; không thay D bằng C dựa trên test.
- A→C tăng F1 test 8,09 điểm phần trăm; B→D tăng 5,93 điểm. A→B tăng 0,18 điểm; C→D giảm 1,98 điểm nhưng tăng Recall 2,76 điểm. Đây là đối chiếu pipeline được tối ưu riêng trên cùng giao thức, chưa chứng minh ưu thế có ý nghĩa thống kê.
- Đã xuất Accuracy, Precision/Recall/F1 nhãn 1, ROC-AUC từ decision_function, TN/FP/FN/TP, tham số, F1 CV mean/SD, các điểm fold và dự đoán từng mẫu. Mỗi ma trận tổng 856, hai hàng có tổng 711 và 145.
- Đã đo riêng tiền xử lý, Grid Search và refit. Suy luận dùng cùng 128 ID, batch 64, 5 lần lặp; có số đo pipeline và toàn luồng. CPU/RAM máy, phiên bản thư viện, kích thước .pkl và thời gian từng lượt được ghi trong [JSON](../results/chapter3_4/experiment_summary.json). Các lượt đo pipeline và toàn luồng là riêng biệt; chênh lệch rất nhỏ giữa hai median không được dùng để suy ra chi phí cleaning.
- Đã đọc và chú giải 8 FP/FN của D: mỗi loại gồm hai mẫu sát ngưỡng và hai mẫu sai xa ngưỡng. Có ID, nhãn thật/dự đoán, score, nội dung và nhận xét định tính trong [ví dụ lỗi](../results/chapter3_4/error_examples.csv), [chú giải](../results/chapter3_4/error_annotations.csv).
- Đã phân tích độ nhạy gần trùng bằng char_wb TF-IDF train-only và cosine ở ngưỡng 0,80/0,90/0,95, không retune. Ngưỡng 0,80 loại 53 mẫu, còn 803; F1 D còn 0,5815. [Bảng độ nhạy](../results/chapter3_4/near_duplicate_sensitivity.csv), [cặp ID gần nhất](../results/chapter3_4/near_duplicate_pairs.csv).

## Hình để dùng

| Mục đích | PNG | SVG |
|---|---|---|
| Slide 16:9, F1 và Recall nhãn 1 | [PNG](figures/abcd_slide_16x9.png) | [SVG](figures/abcd_slide_16x9.svg) |
| A–D, bốn chỉ số | [PNG](figures/abcd_metric_comparison.png) | [SVG](figures/abcd_metric_comparison.svg) |
| Đối chiếu A–C, B–D, A–B, C–D | [PNG](figures/abcd_pairwise_comparison.png) | [SVG](figures/abcd_pairwise_comparison.svg) |
| Ma trận A–D, số đếm | [PNG](figures/abcd_confusion_overview.png) | [SVG](figures/abcd_confusion_overview.svg) |
| Ma trận A–D, chuẩn hóa theo hàng | [PNG](figures/abcd_confusion_overview_normalized.png) | [SVG](figures/abcd_confusion_overview_normalized.svg) |
| Mốc cơ sở/tối ưu có Underthesea | [PNG](figures/metric_comparison.png) | [SVG](figures/metric_comparison.svg) |

Tất cả PNG nền trắng 300 DPI; SVG giữ glyph tiếng Việt bằng đường nét. Đã xem cả PNG và bản render SVG của đủ 20 hình; sửa lỗi pattern thành mảng đen bằng nét vector tường minh và nới lề thang màu chuẩn hóa. Bản slide có canvas đúng 16:9 (4.800×2.700 pixel). Mã sinh ảnh: [evaluation_visuals.py](../src/evaluation_visuals.py); số liệu lấy từ JSON/CSV, không sửa tay giá trị trên hình.

## Ảnh đã thay, thêm và xóa

Đã thay đúng cả `.png` và `.svg` của 14 tên sau (28 tệp):

- confusion_overview
- confusion_svm_linear_baseline_counts
- confusion_svm_linear_baseline_normalized
- confusion_svm_linear_tuned_counts
- confusion_svm_linear_tuned_normalized
- confusion_svm_rbf_baseline_counts
- confusion_svm_rbf_baseline_normalized
- confusion_svm_rbf_tuned_counts
- confusion_svm_rbf_tuned_normalized
- cv_vs_test_f1
- error_breakdown
- evaluation_flow
- metric_comparison
- predicted_label_distribution

Thêm cả PNG và SVG của 6 tên (12 tệp): abcd_confusion_overview, abcd_confusion_overview_normalized, abcd_metric_comparison, abcd_pairwise_comparison, abcd_slide_16x9, confusion_overview_normalized.

**Không xóa ảnh nào** vì toàn bộ tên cũ được giữ lại. Danh sách đầy đủ từng tệp: [figure_changes.json](figure_changes.json). Ảnh tạm đã kiểm tra nằm trong `tmp/visuals_review/`; thư mục tmp bị Git bỏ qua.

## Kiểm tra và giới hạn còn lại

- 36 unit test đều đạt: [test_log.txt](../results/chapter3_4/test_log.txt).
- Kiểm tra tích hợp đạt cho 4 mô hình A–D và 4 mốc cũ: nhãn, ma trận, số liệu CSV/JSON, lựa chọn CV, 400 fit, hash đầu vào/mô hình, PNG 300 DPI, SVG parse/render, giới hạn chữ và liên kết: [qa_checks.json](qa_checks.json).
- Dự đoán và decision score C/D khớp toàn bộ với mốc tối ưu cũ (sai số score ≤ 1e-10).
- Đã mở tệp HTML/README để phân tích và kiểm tra liên kết nội bộ. **Chưa xác nhận giao diện HTML bằng Chrome**: Computer Use tự dừng do không xác định chắc URL hiện tại trên Windows. Không coi kiểm tra liên kết là thay thế cho kiểm tra giao diện trong trình duyệt.
- A–D không còn nhánh thực nghiệm bắt buộc nào thiếu. Chưa có test ngoài theo thời gian/sự kiện/nguồn; holdout nội bộ đã được quan sát ở lần đánh giá trước. Gần trùng mới là độ nhạy từ vựng, chưa phải chia nhóm ngữ nghĩa/sự kiện. Chưa đo RAM mô hình khi chạy; chỉ báo RAM máy và kích thước tệp trên đĩa.
- Các .pkl A–D có thực tại `artifacts/chapter3_4/`; dữ liệu và mô hình nhị phân tiếp tục bị Git bỏ qua theo quy ước sẵn có. Mã nguồn, báo cáo, CSV/JSON, dự đoán và hình được đưa lên Git.

Skill scientific-visualization đã được cài riêng vào Codex từ cùng kho GitHub được yêu cầu. Đã thử lệnh npx.cmd ở DACN; clone bị kẹt nên tải riêng thư mục skill và 17 tệp hỗ trợ qua GitHub API/raw. Đã đọc SKILL.md, áp dụng hướng dẫn và thêm dẫn nguồn Scientific Agent Skills trong báo cáo. [Hồ sơ tái lập và hash nguồn](../results/chapter3_4/reproducibility.json).
