FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml .
COPY app ./app
COPY migrations ./migrations
RUN pip install --no-cache-dir '.[postgres]'
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
