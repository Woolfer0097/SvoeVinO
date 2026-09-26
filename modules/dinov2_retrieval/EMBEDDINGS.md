# Генерация эмбеддингов на GPU

Перенесите на новый компьютер репозиторий и **отдельно** файл
`modules/dinov2_retrieval/data/exports/wine_catalog_bundle.tar` (папка `data/`
игнорируется Git). В архиве находятся дамп PostgreSQL, исходная CSV,
`dataset_photos/` с фотографиями из CSV, `web_photos/` и
`dataset_photo_manifest.csv`. Дамп сделан **до** расчёта эмбеддингов: векторы
пока `NULL`. Для запуска нужны PostgreSQL с pgvector и Python 3.11+ на машине
с NVIDIA GPU.

## Подготовка

```bash
mkdir -p ~/wine-transfer
tar -xf /path/to/wine_catalog_bundle.tar -C ~/wine-transfer
export WINE_BUNDLE_DIR="$HOME/wine-transfer/wine_catalog_bundle"
# Следующие команды выполняются из корня перенесённой копии репозитория.
cd modules/dinov2_retrieval
python3.12 -m venv .venv

# Установите CUDA-сборку PyTorch, подходящую для RTX 5070, по официальному
# селектору https://pytorch.org/get-started/locally/ (CUDA 12.8 или новее).
.venv/bin/pip install -e '.[db,ml]'
.venv/bin/python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
```

В WSL держите архив, распакованные фото и виртуальное окружение в файловой
системе Linux, а не под `/mnt/c`. Изображения читаются по одному, текст по
восемь записей в батче; это ограничивает расход оперативной и видеопамяти.
Для Python, PyTorch и моделей понадобится несколько гигабайт свободного места.

Архив уже содержит только нужные исходные фото, поэтому распаковывать большие
Strapi RAR на новом WSL-компьютере не нужно. Укажите `dataset_photos/` как
`--uploads-root`. Карту `dataset_photo_manifest.csv` можно открыть в таблице:
для `ambiguous` перечислены варианты в `candidate_paths`, а `missing` означает,
что подходящий файл не найден. Чтобы вручную разрешить неоднозначность,
впишите точный путь относительно `uploads` в `relative_path` и поставьте
`status=matched`.

Сейчас автоматически сопоставлены 1 981 из 2 090 уникальных имён фото:
67 неоднозначны и 42 отсутствуют. Поэтому полный проход `dataset_photo` вернёт
частичный результат для этих строк; текст и доступные веб-фото считаются
независимо.

Для отдельной локальной БД можно восстановить дамп в новую базу:

```bash
cp .env.example .env
docker compose up -d postgres
docker compose exec -T postgres createdb -U dinov2 wine_embeddings
docker compose exec -T postgres pg_restore -U dinov2 -d wine_embeddings \
  --no-owner --no-acl < "$WINE_BUNDLE_DIR/wine_catalog.postgres.dump"
export DATABASE_URL='postgresql://dinov2:dinov2@localhost:5433/wine_embeddings'
```

Пароль берётся из `DATABASE_URL`; замените пример значениями своего сервера.
При работе с уже восстановленной БД повторное восстановление не требуется.

## Расчёт

```bash
# Пробные три строки каждого вида (повторный запуск возьмёт следующие NULL).
.venv/bin/wine-pipeline embed --kind description_text --device cuda \
  --text-batch-size 8 --limit 3
.venv/bin/wine-pipeline embed --kind web_photo --device cuda \
  --web-photos-root "$WINE_BUNDLE_DIR/web_photos" --limit 3
.venv/bin/wine-pipeline embed --kind dataset_photo --device cuda \
  --uploads-root "$WINE_BUNDLE_DIR/dataset_photos" \
  --photo-manifest "$WINE_BUNDLE_DIR/dataset_photo_manifest.csv" --limit 3

# Полный проход: команды безопасно перезапускать после остановки.
.venv/bin/wine-pipeline embed --kind description_text --device cuda --text-batch-size 8
.venv/bin/wine-pipeline embed --kind web_photo --device cuda \
  --web-photos-root "$WINE_BUNDLE_DIR/web_photos"
.venv/bin/wine-pipeline embed --kind dataset_photo --device cuda \
  --uploads-root "$WINE_BUNDLE_DIR/dataset_photos" \
  --photo-manifest "$WINE_BUNDLE_DIR/dataset_photo_manifest.csv"
```

Изображения кодирует существующая `facebook/dinov2-small` (384 значения).
Текст из названия, категории, цвета, региона, сорта, описания и винодельни
кодирует `intfloat/multilingual-e5-base` (768 значений); используется префикс
`passage: `, усреднение токенов и L2-нормализация. Это разные пространства
векторов: сравнивать текстовый и визуальный векторы между собой нельзя.
Названия моделей записываются в `*_embedding_model`. Изменение фотографии или
текстовых полей при повторном импорте сбрасывает соответствующий вектор.

Для ошибок есть `wine_embedding_jobs.last_error`. Команды пропускают готовые
векторы, а `--retry-failed` выбирает только ошибочные строки. После ручной
правки карты фотографий запустите `dataset_photo` снова.

```sql
SELECT count(*) FILTER (WHERE dataset_photo_embedding IS NOT NULL) AS dataset,
       count(*) FILTER (WHERE web_photo_embedding IS NOT NULL) AS web,
       count(*) FILTER (WHERE description_text_embedding IS NOT NULL) AS text
FROM wines;
SELECT kind, status, count(*) FROM wine_embedding_jobs GROUP BY kind, status;
```

После генерации создайте новый `pg_dump -Fc`, чтобы передать уже заполненные
векторы.
