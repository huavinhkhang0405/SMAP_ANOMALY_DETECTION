# Kế Hoạch Sprint

Tài liệu theo dõi mục tiêu, công việc và sản phẩm bàn giao cho từng sprint. Sprint 1 và 2 đã hoàn thành.

## Sprint 1 - Nền Tảng Dữ Liệu (Hoàn thành)

### Mục tiêu
- Hiểu một kênh SMAP (P-1).
- Chốt data contract và config.
- Đảm bảo sliding window chạy đúng với shape đã verify.
- Tạo các trực quan cơ sở.

### Công việc
- EDA nhẹ: shape, kiểm tra đơn biến, giá trị thiếu, thang đo, định dạng nhãn.
- Vẽ biểu đồ tín hiệu + histogram + trung bình trượt.
- Triển khai sliding window sequence generation.
- Xác minh output shapes với asserts.
- Lưu processed arrays để tái sử dụng.

### Sản phẩm bàn giao
- config.py frozen values.
- preprocessing.py với create_sequences.
- data/processed/X_train.npy
- data/processed/y_train.npy
- report/eda_*.md và report/eda_*.png
- report/baseline_*.png

## Sprint 2 - Nền Tảng Dự Báo (Hoàn thành)

### Mục tiêu
- Huấn luyện LSTM cho dự báo bước kế tiếp.
- Dùng chia validation theo thời gian (no shuffle).
- Lưu residual artifacts cho chấm điểm bất thường.
- Chốt rolling feature design.

### Công việc
- Xây dựng baseline LSTM(64) với dropout.
- Dùng EarlyStopping + ModelCheckpoint.
- Dự đoán trên test split; compute residuals.
- Lưu y_true, y_pred, residuals, residual_features.
- Vẽ chuỗi residual để thấy tương phản bất thường.

### Sản phẩm bàn giao
- src/model/lstm_forecast.py
- src/utils/residual_features.py
- train.py (huấn luyện + residual artifacts)
- data/processed/y_true.npy
- data/processed/y_pred.npy
- data/processed/residuals.npy
- data/processed/residual_features.npy
- report/residuals.png

## Sprint 3 - Chấm Điểm Bất Thường (Kế hoạch)

### Mục tiêu
- Huấn luyện Isolation Forest trên residual features.
- Chấm điểm bất thường và sinh nhãn.
- Đánh giá với các khoảng có nhãn nếu có.

### Công việc
- Fit Isolation Forest với contamination cố định.
- Tạo anomaly score và nhãn nhị phân.
- Vẽ anomaly score theo thời gian.
- So sánh với anomaly có nhãn cho một kênh.
- Tính các chỉ số phân loại (Precision, Recall, F1-Score).

### Sản phẩm bàn giao
- src/model/isolation_forest.py
- report/anomaly_score.png
- report/anomaly_overlay.png
- report/metrics.md

## Sprint 4 - Tích Hợp Pipeline (Kế hoạch)

### Mục tiêu
- Kết nối preprocessing (preprocessing.py), forecasting (lstm.keras) và scoring (isolation_forest.pkl) vào inference pipeline thống nhất, sạch.
- Cung cấp một hàm/entry point Python duy nhất cho end-to-end inference phục vụ UI.
- Loại bỏ phụ thuộc dữ liệu mock vì assets thật đã có sớm.

### Công việc
- Thêm module pipeline src/pipeline/inference_pipeline.py để xử lý luồng dữ liệu đầy đủ <br>
(CSV $\rightarrow$ Cửa sổ trượt $\rightarrow$ Dự đoán LSTM $\rightarrow$ Residual trượt $\rightarrow$ Isolation Forest $\rightarrow$ Nhãn).
- Thêm config hooks trong src/config.py để chuyển đổi đường dẫn cho mô hình thật và kênh dữ liệu.
- Đảm bảo mọi thành phần tái lập bằng random seed toàn cục cố định.
- Xây dựng dashboard frontend tương tác bằng Streamlit (app/streamlit_app.py) gọi inference pipeline và render lớp phủ bất thường.

### Sản phẩm bàn giao
- src/pipeline/inference_pipeline.py (script tích hợp lõi)
- app/app.py (ứng dụng dashboard web Streamlit)
- report/pipeline_run.md (log và xác minh tích hợp)

## Sprint 5 - Demo và Báo Cáo (Kế hoạch)

### Mục tiêu
- Chuẩn bị tài nguyên demo và báo cáo cuối.
- Tóm tắt kết quả và hạn chế.

### Công việc
- Tạo slide hoặc mục báo cáo ngắn.
- Thêm các biểu đồ và bảng cuối.
- Viết kết luận và hướng phát triển.

### Sản phẩm bàn giao
- report/final_summary.md
- report/final_plots/
