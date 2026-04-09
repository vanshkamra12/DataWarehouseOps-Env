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

# Make the orchestrator script executable
RUN chmod +x run.sh

# Expose port for HF Spaces
EXPOSE 7860

# Run both the server and the inference script
CMD ["/app/run.sh"]
