# SvoeVinO — распознавание вина по фотографии

SvoeVinO принимает фотографию бутылки или этикетки, находит наиболее похожее
вино в каталоге и показывает карточку товара. Результат объединяет визуальный
поиск DINOv2, OCR, геометрическую проверку SuperPoint/LightGlue, цвет и год.

Основной контракт проверки:

```http
POST /v1/eval/predict
Content-Type: multipart/form-data
image=<JPEG|PNG|WebP, до 10 MiB>
```

```json
{"slug":"kokur-suhoe-2025"}
```

## Материалы

- [Архитектура, стек, развертывание и API](docs/PROJECT_DOCUMENTATION.md)
- [Чек-лист и ссылки для формы сдачи](docs/SUBMISSION.md)
- [Общий backend](modules/wine_pipeline/README.md)
- [Frontend](frontend/README.md)
- [DINOv2 retrieval](modules/dinov2_retrieval/README.md)
- [OCR](modules/ocr/README.md)
- [SuperPoint/LightGlue](modules/superpoint/README.md)
- [Предобработка](modules/wine_label_preprocessing/README.md)

## Быстрый запуск через Docker Compose

### Требования

- Git, Docker Engine и Docker Compose v2 с поддержкой `gpus` и Compose override;
- NVIDIA GPU, актуальный драйвер и NVIDIA Container Toolkit;
- от 16 GiB RAM и около 15 GiB места для образов, моделей и данных;
- runtime-архив: дамп PostgreSQL с готовыми эмбеддингами и эталонные фото.

Проверка GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi
```

> Большие данные, дамп и кэши не хранятся в Git. Перед сдачей опубликуйте
> runtime-архив без запроса доступа и внесите ссылку в
> [docs/SUBMISSION.md](docs/SUBMISSION.md). Без него чистый клон не сможет
> выполнить полноценное распознавание.

Все команды выполняются из корня репозитория в Linux/macOS или WSL.

### 1. Получить код и данные

```bash
git clone https://github.com/Woolfer0097/SvoeVinO.git
cd SvoeVinO
mkdir -p modules/dinov2_retrieval/data/reference/feedback
mkdir -p modules/dinov2_retrieval/model-cache
```

Распакуйте публичный runtime-архив в `modules/dinov2_retrieval/data/`.
Ожидаются как минимум `web_photos/`, `reference/catalog/` и дамп в `exports/`.

### 2. Восстановить каталог

```bash
cp modules/dinov2_retrieval/.env.example modules/dinov2_retrieval/.env
docker compose -f modules/dinov2_retrieval/docker-compose.yml up -d postgres
docker compose -f modules/dinov2_retrieval/docker-compose.yml exec -T postgres \
  pg_restore -U dinov2 -d dinov2 --clean --if-exists --no-owner --no-acl \
  < modules/dinov2_retrieval/data/exports/dinov2-giant-e5-20260929.dump
```

Имя дампа может отличаться — используйте имя из опубликованного архива. Если
в дампе ещё нет `reference_images`, опубликуйте готовые векторы из `wines`:

```bash
docker compose -f modules/dinov2_retrieval/docker-compose.yml \
  --profile pipeline run --rm --entrypoint python wine-pipeline \
  -m dinov2_retrieval.wine_catalog.publish_references --data-root /data
```

### 3. Запустить приложение

Основной профиль использует CUDA для DINOv2 и SuperPoint, CPU — для OCR/E5:

```bash
docker compose -f compose.pipeline.yml up -d --build
docker compose -f compose.pipeline.yml ps
docker compose -f compose.pipeline.yml logs -f dinov2 superpoint ocr backend
```

Первый запуск дольше последующих: скачиваются веса моделей. После готовности:

| Компонент | Адрес |
| --- | --- |
| Web-интерфейс | <http://localhost:3000> |
| Swagger backend | <http://localhost:8080/docs> |
| OpenAPI-схема | <http://localhost:8080/openapi.json> |
| PostgreSQL (localhost) | `127.0.0.1:5433` |

### 4. Проверить распознавание

```bash
curl -i -F 'image=@/absolute/path/to/photo.jpg' \
  http://127.0.0.1:8080/v1/eval/predict
```

Асинхронный режим с полной диагностикой:

```bash
curl -F 'image=@/absolute/path/to/photo.jpg' \
  'http://127.0.0.1:8080/v1/eval/predict?wait=false'
curl 'http://127.0.0.1:8080/image/status?job_id=JOB_ID'
```

Остановка без удаления БД и кэшей:

```bash
docker compose -f compose.pipeline.yml stop
docker compose -f modules/dinov2_retrieval/docker-compose.yml stop postgres
```

Не используйте `down -v`, если хотите сохранить базу.

## CPU-only стенд

Создайте игнорируемый Git файл `.env.cloud`:

```dotenv
POSTGRES_PASSWORD=replace-with-a-long-random-password
DATABASE_URL=postgresql://dinov2:replace-with-a-long-random-password@postgres:5432/dinov2
```

```bash
docker compose --env-file .env.cloud \
  -f compose.pipeline.yml -f compose.cloud-cpu.yml up -d --build
```

Frontend будет на порту 80. CPU-профиль существенно медленнее. Для публичного
стенда дополнительно нужны HTTPS, firewall и reverse proxy.

## Тесты

Python-модули тестируются независимо. Пример:

```bash
cd modules/wine_pipeline
python3.12 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest -q
```

Аналогично запускаются тесты в остальных каталогах `modules/*`. Unit-тесты не
требуют сети, GPU или весов; retrieval integration-тесты запускаются отдельно.

Frontend:

```bash
cd frontend
npm ci
npm run typecheck
npm run build
```

## Структура

```text
frontend/                              Nuxt 4 / Vue 3
modules/wine_pipeline/                 оркестратор и публичный API
modules/wine_label_preprocessing/      visual/OCR предобработка
modules/dinov2_retrieval/              DINOv2, каталог, pgvector
modules/superpoint/                    геометрическая проверка
modules/ocr/                           PaddleOCR и E5
docs/                                  документация сдачи
compose.pipeline.yml                   основной профиль
compose.pipeline.gpu.yml               all-GPU override
compose.cloud-cpu.yml                  CPU-only override
```

## Ограничения

- score — ранговый сигнал, а не вероятность правильного ответа;
- очередь хранится в памяти одного backend worker;
- первая обработка медленнее из-за загрузки моделей;
- качество оценивается только на отдельной размеченной выборке;
- публичный deployment требует TLS, rate limiting и уникальные секреты.

Лицензии моделей могут отличаться от лицензии кода. Веса SuperPoint имеют
отдельные условия Magic Leap — проверьте лицензии перед коммерческим применением.
