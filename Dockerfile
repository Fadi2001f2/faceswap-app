FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir opencv-python onnxruntime pillow tqdm gdown basicsr insightface fastapi uvicorn python-multipart

RUN mkdir -p backend/models
RUN python -c "import urllib.request, os; \
    os.path.exists('backend/models/inswapper_128.onnx') or urllib.request.urlretrieve('https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx', 'backend/models/inswapper_128.onnx'); \
    os.path.exists('backend/models/GFPGANv1.4.pth') or urllib.request.urlretrieve('https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth', 'backend/models/GFPGANv1.4.pth')"

EXPOSE 7860
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
