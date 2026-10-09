FROM python:3.11-slim

WORKDIR /app

# Системные зависимости для PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Python-зависимости
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Код
COPY backend /app/backend
COPY frontend /app/frontend

# Директории для кэша и загрузок
RUN mkdir -p /app/backend/cache /app/backend/uploads

WORKDIR /app/backend
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]