# Документация SvoeVinO

## Назначение и архитектура

SvoeVinO распознаёт вино по фотографии и возвращает slug карточки каталога.
Расширенный API также отдаёт Top-10, диагностику и предупреждение о
неуверенном совпадении.

```text
Браузер / проверяющий клиент
        │
        ├── Nuxt frontend :3000
        │        │ server-side proxy
        │        ▼
        └── Wine Pipeline API :8080
                  ├── Preprocessing ── visual + OCR images
                  ├── DINOv2 giant ────────────────┐
                  ├── SuperPoint + LightGlue       ├── fusion → color/year → Top-10
                  └── PaddleOCR + multilingual E5 ┘             │
                                │                                ▼
                                └── PostgreSQL 16 + pgvector → slug/card
```

Сервисы связаны по HTTP и не импортируют код друг друга. ML-сервисы доступны
только внутри Docker-сети; наружу публикуются frontend и общий backend.

Поток запроса:

1. Backend проверяет тип, размер (до 10 MiB) и число пикселей.
2. Предобработка применяет EXIF-ориентацию и готовит две версии изображения.
3. DINOv2 и OCR/E5 работают параллельно.
4. SuperPoint/LightGlue геометрически проверяет визуальных кандидатов.
5. Оркестратор объединяет ветви по `slug`, учитывает палитру и прочитанный год.
6. Добровольный feedback может добавить подтверждённое фото в индекс.

## Стек и компоненты

| Слой | Технологии |
| --- | --- |
| Frontend | Nuxt 4, Vue 3, TypeScript, Nitro |
| Backend | Python 3.11+, FastAPI, Uvicorn, HTTPX |
| Изображения | OpenCV, Pillow, NumPy |
| Visual retrieval | DINOv2 with registers giant, PyTorch, Transformers |
| Геометрия | SuperPoint, LightGlue, OpenCV RANSAC |
| Текст | PaddleOCR, multilingual-e5-base |
| Данные | PostgreSQL 16, pgvector, файловое хранилище |
| Инфраструктура | Docker Compose, NVIDIA Container Toolkit |
| Тесты | pytest, Playwright, vue-tsc |

| Компонент | Ответственность | Подробнее |
| --- | --- | --- |
| `frontend` | загрузка, polling, карточка, feedback | [README](../frontend/README.md) |
| `wine_pipeline` | очередь, fusion, публичный API | [README](../modules/wine_pipeline/README.md) |
| `wine_label_preprocessing` | варианты предобработки | [README](../modules/wine_label_preprocessing/README.md) |
| `dinov2_retrieval` | embedding и pgvector-поиск | [README](../modules/dinov2_retrieval/README.md) |
| `superpoint` | matching и гомография | [README](../modules/superpoint/README.md) |
| `ocr` | OCR и текстовый поиск | [README](../modules/ocr/README.md) |

## Публичный API

Swagger доступен по `/docs`, OpenAPI — по `/openapi.json`.

### `POST /v1/eval/predict`

Принимает multipart-поле `image` (JPEG, PNG или WebP). Без параметров ждёт
завершения и возвращает `{"slug":"..."}`. С `?wait=false` возвращает HTTP 202,
`job_id` и интервал polling. Заголовок `X-Job-ID` синхронного ответа позволяет
запросить полную диагностику.

### `GET /image/status?job_id=...`

Возвращает `queued`, `processing`, `done` или `failed`, текущий этап, прогресс,
историю и полный результат. Неизвестная/истёкшая задача — 404.

### `POST /feedback`

Принимает оценку завершённого результата:

```json
{
  "job_id": "UUID",
  "is_correct": false,
  "correct_slug": "expected-slug",
  "wine_information": {
    "name": "Название",
    "year": "2021",
    "locality": "регион",
    "winery": "винодельня",
    "notes": "дополнительные сведения"
  }
}
```

Также есть `GET /feedback/status?job_id=...` и `GET /feedback/stats`.

| Код | Значение |
| ---: | --- |
| 200 | запрос обработан |
| 202 | задача поставлена в очередь |
| 400/413/415/422 | неверный/слишком большой файл или запрос |
| 404 | задача не найдена или истекла |
| 429 | очередь заполнена |
| 503 | обязательная ветвь или каталог недоступны |

Точная схема конкретной версии стенда всегда находится в Swagger.

## Данные, модели и развертывание

Исходный экспорт содержит 4 147 строк и 2 103 уникальных slug. Для работы нужны
каталог и embeddings в PostgreSQL, `reference_images`, эталонные фото и веса
моделей. Большие данные, дампы и кэши исключены из Git. Для воспроизводимости
их нужно опубликовать отдельным архивом с SHA-256 и открытым доступом. Архив не
должен содержать секреты, cookies или пользовательские фото без разрешения.

Пошаговый запуск дан в [корневом README](../README.md):

- `compose.pipeline.yml` — DINO/SuperPoint на GPU, OCR/E5 на CPU;
- `compose.pipeline.gpu.yml` — all-GPU override;
- `compose.cloud-cpu.yml` — CPU-only профиль одной VM;
- PostgreSQL запускается compose-файлом retrieval-модуля.

Production должен добавить HTTPS, firewall, rate limiting, мониторинг,
резервное копирование и управление секретами. Backend использует один Uvicorn
worker, поскольку очередь задач хранится в памяти. Для нескольких реплик нужны
внешняя очередь и общее хранилище состояния.

## Тестирование и ограничения

Unit-тесты модулей независимы от сети, GPU и реальных весов. End-to-end smoke
требует работающий стек и размеченное командное фото:

```bash
cd modules/wine_pipeline
.venv/bin/python -m wine_pipeline.smoke \
  --image /path/to/validation.webp --expected-slug EXPECTED_SLUG \
  --output reports/smoke.json
```

Контрольные фото организатора используются только для финальной проверки
протокола, не для подбора весов.

- score не является калиброванной вероятностью;
- качество зависит от каталога, ракурса, бликов и читаемости этикетки;
- автоматическая локализация/развёртка без внешней маски не заявляется;
- асинхронные задачи теряются при рестарте backend;
- CPU-only режим имеет высокую задержку;
- лицензии кода, моделей, весов и данных проверяются отдельно.
