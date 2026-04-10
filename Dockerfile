FROM python:3.11-slim

# Metadata
LABEL maintainer="vanshkamra27@gmail.com"
LABEL description="DataWarehouseOps-Env: OpenEnv-compliant RL environment for data engineering tasks"

# HuggingFace Spaces runs on port 7860
ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Copy requirements first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project
COPY . .

# Expose port for HF Spaces
EXPOSE 7860

# Uvicorn with auto-reload disabled (production mode)
# HF Spaces expects the app on 0.0.0.0:7860
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
