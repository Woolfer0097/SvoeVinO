FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .
USER 1000:1000
EXPOSE 8080
CMD ["uvicorn", "--factory", "wine_pipeline.api:create_app", "--host", "0.0.0.0", "--port", "8080"]
