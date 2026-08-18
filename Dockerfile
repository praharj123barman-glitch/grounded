FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY grounded ./grounded
COPY data/sample ./data/sample

EXPOSE 8000
# Bind to $PORT when the platform sets one (Render, HF Spaces), else 8000.
CMD ["sh", "-c", "uvicorn grounded.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
