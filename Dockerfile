FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY grounded ./grounded
COPY data/sample ./data/sample

EXPOSE 8000
CMD ["uvicorn", "grounded.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
