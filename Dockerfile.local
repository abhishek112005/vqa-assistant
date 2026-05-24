FROM python:3.11-slim

WORKDIR /app

COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

COPY api.py .

EXPOSE 7860

# Render injects $PORT; fall back to 7860 for local Docker runs
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-7860}"]
