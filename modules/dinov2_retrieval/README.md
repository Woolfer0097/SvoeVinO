# DINOv2 Retrieval

Изолированный модуль получения и проверки фотографий с генерацией embedding
через `facebook/dinov2-small`. PostgreSQL, pgvector, OCR, Airflow, SuperPoint,
LightGlue и frontend пока не подключены.

## Структура модулей

```text
src/dinov2_retrieval/
├── __init__.py
├── config.py
├── contracts.py
├── cli.py                         # совместимость с python -m ...cli
├── entrypoints/
│   ├── __init__.py
│   └── cli.py                     # разбор аргументов CLI
├── application/
│   ├── __init__.py
│   ├── validate_image.py          # сценарий проверки изображения
│   └── create_embedding.py        # сценарий создания embedding
├── preprocessing/
│   ├── __init__.py
│   └── image_preprocessor.py      # Pillow, decode и перевод в RGB
├── embedding/
│   ├── __init__.py
│   ├── base.py                    # общий Protocol embedder
│   └── dinov2_embedder.py         # DINOv2-small
├── retrieval/
│   ├── __init__.py
│   └── base.py                    # Protocol будущего поиска
└── infrastructure/
    ├── __init__.py
    └── storage/
        ├── __init__.py
        └── local_storage.py       # LocalImageStorage и проверки пути
```

`storage.py` в корне пакета оставлен тонким compatibility-фасадом: старые
импорты `dinov2_retrieval.storage` продолжают работать, но реализация находится
только в `infrastructure/storage/local_storage.py`.

`retrieval` содержит только интерфейс. DINOv2 реализован через optional ML-
зависимости, а PostgreSQL, pgvector и реальный search-backend пока не
подключены.

## Конфигурация

Модуль получает корень данных из обязательной переменной `DATA_ROOT`.
Поддерживаемые расширения задаются через `SUPPORTED_IMAGE_EXTENSIONS`.
По умолчанию поддерживаются `.jpg`, `.jpeg`, `.png`, `.webp`.

`MAX_IMAGE_SIZE_BYTES` задаёт максимальный размер файла. По умолчанию —
10 MiB.

`HF_HOME` задаёт каталог кэша весов Hugging Face.

Пример настройки:

```bash
cp .env.example .env
export DATA_ROOT=/data
export MAX_IMAGE_SIZE_BYTES=10485760
export SUPPORTED_IMAGE_EXTENSIONS=.jpg,.jpeg,.png,.webp
export HF_HOME=/root/.cache/huggingface
```

`SUPPORTED_IMAGE_EXTENSIONS` — список расширений через запятую. Разрешены
только `jpg`, `jpeg`, `png` и `webp`.

## Установка и запуск

```bash
cd modules/dinov2_retrieval
python -m venv .venv
.venv/bin/pip install -e '.[dev]'

DATA_ROOT=/data .venv/bin/python -m dinov2_retrieval.cli validate \
  --image-uri /data/queries/test.jpg
```

Команда выводит JSON с путём проверенного файла, его шириной, высотой и MIME-
типом. Для локального запуска `embed` с настоящей моделью дополнительно нужны
ML-зависимости: `.venv/bin/pip install -e '.[dev,ml]'`.

## Запуск в Docker

Docker Compose содержит только сервис `dinov2-retrieval`. Он запускает текущий
CLI-модуль; отдельного HTTP-сервера пока нет.

Подготовить локальную директорию с данными и собрать образ:

```bash
cd modules/dinov2_retrieval
cp .env.example .env
mkdir -p data/queries
mkdir -p model-cache
docker compose build
```

Compose подключает `.env` через `env_file`. `DATA_ROOT` внутри контейнера
остаётся равным `/data`, потому что это путь, куда монтируется локальная
директория `./data`. Лимит `MAX_IMAGE_SIZE_BYTES` и `HF_HOME` берутся из
`.env`/Compose.

При первом запуске `embed` будут скачаны веса модели
`facebook/dinov2-small`. Они сохраняются в локальный `./model-cache` и не
скачиваются заново при последующих запусках.

Проверить изображение из локальной `./data`:

```bash
docker compose run --rm dinov2-retrieval validate \
  --image-uri /data/queries/test.jpg
```

Получить embedding:

```bash
docker compose run --rm dinov2-retrieval \
  embed --image-uri /data/queries/test.jpeg
```

Команда `embed` выводит только первые пять значений embedding. Полный список
из 384 чисел возвращается внутри application-сценария
`create_embedding.py`.

Для просмотра команды запуска Compose:

```bash
docker compose up
```

По умолчанию она выводит справку CLI и завершает контейнер с кодом `0`.
Healthcheck проверяет импорт модуля и доступность `DATA_ROOT=/data`; так как
текущий модуль является одноразовой CLI-командой, healthcheck не превращает его
в постоянно работающий сервис.

Запустить тесты внутри собранного образа:

```bash
docker compose run --rm dinov2-retrieval pytest
```

Каталог `./data` монтируется в контейнер только для чтения как `/data`.
Каталог `./model-cache` монтируется в `/root/.cache/huggingface`.
PostgreSQL, Redis, Airflow, frontend и другие сервисы в Compose не добавляются.

## Тесты

```bash
cd modules/dinov2_retrieval
.venv/bin/pytest
```

Тесты используют mock-модель и mock-процессор для DINOv2, поэтому обычный
`pytest` не скачивает веса из интернета и не требует ML-зависимостей в локальном
`.venv`.
