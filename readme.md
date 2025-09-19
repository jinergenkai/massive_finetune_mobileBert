#MASSIVE Fine-tuning for mobile devices using BERT 
## 1. Dữ liệu đầu vào
Dataset: MASSIVE Dataset từ AmazonScience/massive.
Ngôn ngữ: Có thể tùy chọn, mặc định là en-US.
Dữ liệu được chia:
- Train: dataset['train']
- Validation: dataset['validation']
- Test: dataset['test'].
## 2. Mô hình sử dụng
Model: MobileBERT (google/mobilebert-uncased).
Số lớp đầu ra: Tương ứng với số lượng intent classes trong dữ liệu.
## 3. Các bước xử lý và huấn luyện
Chuẩn bị dữ liệu:
Tokenize các câu bằng MobileBertTokenizer.
Ánh xạ intent thành nhãn số hóa (label_encoder).
Cấu hình huấn luyện:
Sử dụng Trainer từ thư viện transformers.
Các tham số chính:
Epochs: Mặc định là 2.
Batch size: Mặc định là 16.
Learning rate: 5e-5.
Warmup steps: 100.
Weight decay: 0.01.
Sử dụng GPU nếu khả dụng.
Huấn luyện:
Huấn luyện trên tập train.
Đánh giá trên tập validation sau mỗi epoch.
Đánh giá:
Tính độ chính xác (accuracy) trên tập test.
## 4. Kết quả
Accuracy: Được in ra sau khi đánh giá trên tập test.
Model và tokenizer: Lưu tại thư mục ./models/mobilebert-massive-quick.
Label encoder: Lưu dưới dạng file label_encoder.pkl.
## 5. Inference nhanh
Có thể kiểm tra nhanh với một số câu mẫu để dự đoán intent và confidence.
## 6. Cấu hình mặc định
Output directory: ./models/mobilebert-massive-quick.
Max sequence length: 128.




## 7. Setup
 docker run --rm -it --gpus=all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark14:22:11

nvidia-smi

pip install  --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
