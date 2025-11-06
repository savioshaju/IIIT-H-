FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# --- install only required system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# --- install minimal Python deps
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
        torch==2.5.1+cpu \
        torchaudio==2.5.1+cpu \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
        Flask==3.0.3 \
        gunicorn==23.0.0 \
        soundfile==0.12.1 \
        numpy==1.26.4 \
        scipy==1.12.0

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
