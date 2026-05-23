FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Copy pre-built ChromaDB (run embed_and_index.py before docker build)
# The data/chroma_db folder is included in the COPY above

EXPOSE 8000

ENV PYTHONPATH=/app
ENV APP_ENV=production

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
