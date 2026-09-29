FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir '.[api]' && useradd --create-home --uid 1000 app
USER app
EXPOSE 8000
CMD ["uvicorn", "wine_label_preprocessing.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
