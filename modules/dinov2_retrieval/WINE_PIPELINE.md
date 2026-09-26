# Wine data pipeline

Пайплайн импортирует все строки CSV в `wines` (повторы slug сохраняются), затем
обрабатывает каждый уникальный slug и связывает одну эталонную фотографию со
всеми его строками. Названия и характеристики берутся из CSV и не меняются
при скрейпинге. Векторы остаются `NULL`. Миграции применяются командой
импорта. Используются `DATABASE_URL` из окружения, существующие Compose и
pgvector настройки модуля.

```bash
cd modules/dinov2_retrieval
cp .env.example .env
mkdir -p data/web_photos
docker compose up -d postgres
docker compose run --rm --build wine-pipeline import-csv /input/strapi_output0709.csv

# Сначала можно проверить несколько slug; полный запуск продолжит незавершённое.
docker compose run --rm --build wine-pipeline scrape --limit 3 --photos-dir /data/web_photos
docker compose run --rm wine-pipeline scrape --photos-dir /data/web_photos
docker compose run --rm wine-pipeline scrape --retry-failed --photos-dir /data/web_photos

# Проверка повторного импорта, дублей slug и общей эталонной фотографии.
docker compose run --rm --entrypoint pytest wine-pipeline \
  -m integration tests/integration/test_wine_catalog_integration.py -q
```

Повторный импорт того же пути обновляет строки по `(import_source,
source_row_number)` и не использует slug как ключ. Для явной идентичности источника
есть `--source-id`. Чтобы проверить выбранные вина, передайте
`--slugs slug-1,slug-2`. Карточки загружаются последовательными HTTP-запросами,
по умолчанию с паузой в секунду, таймаутом 20 секунд и тремя повторами с
экспоненциальной задержкой. Статус, число попыток и последняя ошибка хранятся в
`wine_scrape_jobs`; `--retry-failed` выбирает только slug со статусом `failed`.

Прогресс полного запуска виден в PostgreSQL, даже если окно команды закрыто:

```bash
docker compose exec -T postgres psql -U dinov2 -d dinov2 -c \
  "SELECT status, count(*) FROM wine_scrape_jobs GROUP BY status ORDER BY status;"
docker compose exec -T postgres psql -U dinov2 -d dinov2 -c \
  "SELECT count(*) AS csv_rows,
          count(*) FILTER (WHERE web_photo IS NOT NULL) AS rows_with_photo,
          count(DISTINCT web_photo) AS downloaded_photos FROM wines;"
```

`pending` ещё не выбран, `in_progress` обрабатывается, `succeeded` получил фото,
`failed` содержит причину в `last_error`. После остановки повторный `scrape`
подхватит все slug, кроме `succeeded`.

Фотографии лежат в `data/web_photos/`, а `web_photo` хранит путь относительно
этой папки. В Compose путь задан через `WINE_PHOTOS_DIR=/data/web_photos`,
поэтому можно опустить `--photos-dir`. Загрузка проверяет, что ответ
декодируется как изображение. Если
сайт запросит CAPTCHA или подтверждение возраста, на хосте установите
`.[db,catalog,catalog-browser]` и Chromium (`playwright install chromium`),
затем запустите `wine-pipeline scrape --interactive` из терминала. Chromium
откроется с сохраняемым профилем; после ручного прохождения можно продолжить
с Enter. Профиль и cookie-файл остаются в игнорируемой Git папке `data/`.

Отчёт `import-csv` печатает количество исходных строк и уникальных slug рядом с
ожидаемыми 2 109. Скрейпинг начинается отдельной командой, после успешного
импорта.

Для переноса базы и фото на другой компьютер и расчёта трёх векторов на GPU
используйте [EMBEDDINGS.md](EMBEDDINGS.md). Команда `photo-manifest` строит
карту исходных имён CSV к файлам Strapi, а `stage-photos` собирает нужные
исходные изображения для компактного архива.

```bash
docker compose run --rm wine-pipeline photo-manifest \
  --uploads-root /input/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads \
  --output /data/exports/dataset_photo_manifest.csv
docker compose run --rm wine-pipeline stage-photos \
  --uploads-root /input/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads \
  --photo-manifest /data/exports/dataset_photo_manifest.csv \
  --output /data/exports/dataset_photos
```
