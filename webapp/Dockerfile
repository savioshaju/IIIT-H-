# ============================
# Render-ready Flask + Torch App
# ============================

# Use a lightweight Python base image
FROM python:3.10-slim

# Prevent Python from writing pyc files and enable real-time logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system-level dependencies (includes ffmpeg + audio libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Upgrade pip and install dependencies
# Install CPU-only Torch build explicitly from the official PyTorch index
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch==2.5.1+cpu torchaudio==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Expose port for Render
EXPOSE 5000

# Use Gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]

