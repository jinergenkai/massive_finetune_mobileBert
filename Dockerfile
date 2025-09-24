# Base image Python 3.13.5 slim
FROM python:3.10-slim

# Tạo thư mục làm việc
WORKDIR /app

# Cài các công cụ cần thiết để build Python package
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl unzip build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy file requirements vào container
COPY reqs_converter.txt .

# Cài đặt package từ requirements file
RUN pip install --no-cache-dir -r reqs_converter.txt

# Copy toàn bộ code vào container
COPY . .

# Lệnh mặc định: chạy script convert
CMD ["python", "auto_convert_pipeline.py", "jinergenkai/mobile-intent-bert"]
