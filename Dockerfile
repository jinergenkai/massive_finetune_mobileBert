FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime

RUN apt-get update && apt-get install -y git wget unzip

WORKDIR /workspace

COPY requirements.txt .
RUN pip install -r requirements.txt
