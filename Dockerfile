FROM python:3.11-slim

LABEL maintainer="openenv-sql-debugger"
LABEL org.opencontainers.image.title="SQL Debugger OpenEnv"
LABEL org.opencontainers.image.description="Real-world SQL debugging environment for AI agents"
LABEL space_sdk="docker"
LABEL tags="openenv"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user

# HF Spaces uses port 7860
EXPOSE 7860

ENV PORT=7860
ENV HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
