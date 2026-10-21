# Multi-Stage Production Dockerfile (Render & Hugging Face Spaces Ready)
FROM python:3.11-slim AS builder

WORKDIR /build
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels -r requirements.txt

FROM python:3.11-slim AS runtime

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=7860

RUN groupadd -r copilot && useradd -r -g copilot -m copilot

COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

COPY backend ./backend
COPY frontend ./frontend
COPY tests ./tests

RUN chown -R copilot:copilot /app
USER copilot

EXPOSE 7860 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python3 -c "import backend.main; print('OK')" || exit 1

CMD ["streamlit", "run", "frontend/streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0"]
