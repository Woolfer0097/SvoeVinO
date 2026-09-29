# SuperPoint verification

Изолированный модуль проверки пары фотографий. Локальные признаки извлекает
SuperPoint, соответствия строит LightGlue из репозитория
[cvg/LightGlue](https://github.com/cvg/LightGlue). Совпадение принимается
только если достаточно соответствий согласованы одной гомографией (RANSAC).

HTTP-сервис читает эталонные фотографии из PostgreSQL (`reference_images`)
и перепроверяет список кандидатов DINOv2. Общий backend связывает этот сервис
с DINOv2 и OCR по HTTP; межмодульных Python-импортов нет.

## Структура

```text
src/superpoint/
├── config.py
├── contracts.py
├── cli.py
├── entrypoints/
│   ├── api.py                      # FastAPI + Swagger UI
│   └── cli.py
├── application/
│   ├── validate_image.py
│   └── verify_photos.py
├── preprocessing/image_preprocessor.py
├── matching/
│   ├── base.py
│   └── superpoint_lightglue.py   # cvg/LightGlue SuperPoint + LightGlue
├── verification/
│   ├── decision.py
│   └── homography.py             # OpenCV RANSAC
└── infrastructure/storage/
    ├── local_storage.py
    └── upload_storage.py           # временные загрузки HTTP API
```

Оба пути должны лежать внутри `DATA_ROOT`. Модуль не ходит в сеть за
изображениями и не пишет результат на диск: CLI печатает JSON.

## Конфигурация

Обязательна переменная `DATA_ROOT`. Остальные значения имеют рабочие
значения по умолчанию.

| Переменная | По умолчанию | Смысл |
| --- | --- | --- |
| `MAX_IMAGE_SIZE_BYTES` | `10485760` | максимальный размер файла |
| `SUPPORTED_IMAGE_EXTENSIONS` | `.jpg,.jpeg,.png,.webp` | допустимые расширения |
| `DEVICE` | `auto` | `auto`, `cpu` или `cuda` |
| `MAX_NUM_KEYPOINTS` | `2048` | лимит точек SuperPoint; `none` снимает лимит |
| `SUPERPOINT_RESIZE` | `1024` | длинная сторона перед детектором; `none` отключает resize |
| `LIGHTGLUE_DEPTH_CONFIDENCE` | `0.95` | ранний выход LightGlue; `-1` отключает |
| `LIGHTGLUE_WIDTH_CONFIDENCE` | `0.99` | отсечение точек; `-1` отключает |
| `MATCH_FILTER_THRESHOLD` | `0.1` | порог уверенности соответствия |
| `MIN_MATCHES` | `15` | минимум соответствий LightGlue |
| `MIN_INLIERS` | `12` | минимум inlier-ов гомографии |
| `MIN_INLIER_RATIO` | `0.3` | доля inlier-ов среди соответствий |
| `RANSAC_REPROJ_THRESHOLD` | `5.0` | допуск RANSAC в пикселях исходного фото |
| `TORCH_HOME` | кэш PyTorch | каталог весов SuperPoint и LightGlue |

Пороги рассчитаны как стартовые для проверки этикетки или бутылки, а не как
откалиброванный production-порог. Гомография хорошо описывает почти плоскую
этикетку и хуже описывает сильный изгиб цилиндра.

Веса SuperPoint распространяются по отдельной ограничительной лицензии
Magic Leap. Код и веса LightGlue — Apache-2.0.

## Установка и запуск

```bash
cd modules/superpoint
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pip install -e '.[ml]'

DATA_ROOT=/data .venv/bin/python -m superpoint.cli verify \
  --query-uri /data/queries/photo.jpg \
  --reference-uri /data/references/label.jpg
```

`validate` только проверяет один файл и не загружает модели:

```bash
DATA_ROOT=/data .venv/bin/python -m superpoint.cli validate \
  --image-uri /data/queries/photo.jpg
```

`verify` завершается с кодом `0`, если проверка выполнена, даже когда пара
не совпала. Признак совпадения — поле `verified` в JSON. Код `1` означает
ошибку пути, формата или рантайма.

Пример ответа:

```json
{
  "verified": true,
  "model": "superpoint+lightglue",
  "device": "cpu",
  "num_matches": 86,
  "num_inliers": 71,
  "inlier_ratio": 0.825581
}
```

При первом `verify` веса скачиваются в `TORCH_HOME` с GitHub Releases
`cvg/LightGlue`.

## HTTP API и Swagger

Сервис слушает только `127.0.0.1`. Порт на хосте — `API_PORT` (по умолчанию
8001), внутри контейнера — 8000, чтобы не пересечься с DINOv2 на 8000.

```bash
cd modules/superpoint
docker compose up -d
docker compose ps    # дождаться healthy; первый старт качает веса и может быть дольше 120 с
```

Откройте <http://localhost:8001> — редирект на Swagger UI. «Try it out» уже
включён. Низкий `score` приходит с кодом 200: сравнение дошло до конца.

`POST /verify` принимает фото запроса и JSON-массив `wine_id` (не больше 20).
Эталоны читаются из PostgreSQL `dinov2` (`127.0.0.1:5433` на хосте,
логин/пароль/база `dinov2`): таблица `reference_images`, колонки `wine_id` и
`image_uri`. Из контейнера этот порт доступен как `host.docker.internal:5433`,
потому что `127.0.0.1` внутри контейнера — сам контейнер. Файлы по
`image_uri` (`/data/reference/...`) берутся из каталога DINOv2: compose
монтирует `modules/dinov2_retrieval/data` целиком в `/data`.

`score` — доля inlier после RANSAC, от 0 до 1, но только если пара прошла
пороги. Иначе 0. Если у id несколько фото, берётся лучшая доля. Ответ
отсортирован по score по убыванию.

| Раздел | Метод и путь | Что делает |
| --- | --- | --- |
| Проверка | `POST /verify` | файл `query` и поле `candidates` (JSON-массив id). Фото не сохраняется. Ответ: `id`, `score` |
| Служебное | `GET /health` | модель загружена: `model` и `device` (Docker healthcheck) |

```bash
curl -s -X POST http://localhost:8001/verify \
  -F query=@photo.jpg \
  -F 'candidates=["shepot","agrolayn_mountain_eagle_cabernet_sauvignon_kaberne_sovinon_krasnoe_suhoe_135_c2ea2996ff"]'
```

Локально без Docker, после `pip install -e '.[api,ml]'`:

```bash
DATA_ROOT=/path/to/dinov2_retrieval/data \
DATABASE_URL=postgresql://dinov2:dinov2@127.0.0.1:5433/dinov2 \
  .venv/bin/uvicorn --factory superpoint.entrypoints.api:create_app \
  --host 127.0.0.1 --port 8001
```

## Запуск в Docker

Compose поднимает долгоживущий сервис `superpoint` командой `serve`.
Разовый CLI по-прежнему через `docker compose run`: тот режим игнорирует
`restart`.

```bash
cd modules/superpoint
cp .env.example .env
mkdir -p data/queries data/references model-cache
docker compose build
```

В `/data` монтируется только для чтения `modules/dinov2_retrieval/data`,
чтобы пути из базы открывались как есть. `./model-cache` монтируется
в `/home/app/.cache/torch` (`TORCH_HOME`), поэтому веса не скачиваются заново.
На Linux каталог кэша должен быть доступен на запись `APP_UID` (по умолчанию 1000).

```bash
docker compose run --rm superpoint verify \
  --query-uri /data/queries/photo.jpg \
  --reference-uri /data/references/label.jpg
```

```bash
docker compose run --rm superpoint pytest
```

Образ — Python 3.12, CPU-колёса PyTorch 2.4.1 и `opencv-python-headless`.
Контейнер не требует NVIDIA runtime и libGL. `DEVICE=cuda` на этом образе
не заработает: для GPU нужна отдельная сборка с CUDA-колёсами.

`docker compose up -d` запускает HTTP API и сам перезапускает контейнер,
пока его не остановят `docker compose stop` / `down`. Healthcheck бьёт в
`GET /health` на порту `PORT` (по умолчанию 8000) и проходит только после
загрузки модели. Первый старт может занять больше 120 с: веса LightGlue
скачиваются до того, как `/health` начнёт отвечать. Пока идёт загрузка,
контейнер может быть `unhealthy`, но `restart` из-за этого его не
останавливает — дождитесь `healthy`. Дальше веса берутся из `model-cache`.

## Тесты

```bash
cd modules/superpoint
.venv/bin/pytest
```

Обычный `pytest` не скачивает веса и не требует ML-зависимостей: матчер и
RANSAC подменяются. Тесты HTTP API пропускаются без extra `api` (и `httpx`
из `dev`). Реальный SuperPoint+LightGlue используется только командой
`verify` или `serve` без внедрённого матчера.
# GPU integration

Set `DEVICE=cuda`, use CUDA-enabled Torch wheels and pass a GPU into Docker.
The common stack is configured in the root `compose.pipeline.yml`.
The pinned LightGlue utility copies results GPU→CPU asynchronously;
the matcher synchronizes CUDA before converting them to Python lists.
