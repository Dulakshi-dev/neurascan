FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Heavy, slow-changing layer first — kept isolated from requirements.txt
# so future edits to that file don't force this to re-download.
RUN pip install --no-cache-dir --timeout=1000 --retries=10 \
    "torch>=2.2,<3.0" "torchvision>=0.17,<1.0" \
    --index-url https://download.pytorch.org/whl/cpu

# Rest of the dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# HuggingFace Spaces runs on port 7860
ENV PORT=7860
EXPOSE 7860

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]