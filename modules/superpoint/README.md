# SuperPoint verification

Изолированный модуль проверки пары фотографий. Локальные признаки извлекает
SuperPoint, соответствия строит LightGlue из репозитория
[cvg/LightGlue](https://github.com/cvg/LightGlue). Совпадение принимается
только если достаточно соответствий согласованы одной гомографией (RANSAC).

PostgreSQL, pgvector, OCR, Airflow, DINOv2 и frontend не подключены.

## Структура

```text
src/superpoint/
├── config.py
├── contracts.py
├── cli.py
├── entrypoints/cli.py
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
└── infrastructure/storage/local_storage.py
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

## Запуск в Docker

Compose содержит только сервис `superpoint`. Отдельного HTTP-сервера нет.

```bash
cd modules/superpoint
cp .env.example .env
mkdir -p data/queries data/references model-cache
docker compose build
```

`./data` монтируется только для чтения в `/data`. `./model-cache` монтируется
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

`docker compose up` печатает справку CLI и завершается с кодом `0`.
Healthcheck проверяет импорт модуля и каталог `DATA_ROOT=/data`.

## Тесты

```bash
cd modules/superpoint
.venv/bin/pytest
```

Обычный `pytest` не скачивает веса и не требует ML-зависимостей: матчер и
RANSAC подменяются. Реальный SuperPoint+LightGlue используется только
командой `verify` без внедрённого матчера.
