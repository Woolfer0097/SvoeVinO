# DINOv2 Retrieval

Сервис визуального поиска вина по фотографии: по снимку бутылки или
этикетки он находит в каталоге **Top-K самых похожих разных вин**. Сравнение
чисто визуальное: фото превращается в вектор из 384 чисел моделью
`facebook/dinov2-small`, и ищутся ближайшие векторы эталонных фотографий в
PostgreSQL с расширением pgvector.

Сервис отвечает только за визуальный поиск. OCR, SuperPoint/LightGlue,
frontend, рекомендации, дообучение DINOv2 и обработка текста — внешние
модули. Кроп, поворот и развёртку этикетки делает `wine_label_preprocessing`;
этот сервис получает уже подготовленное изображение.

## Как работает модуль

### Три сценария

```text
index     data/reference ──► для каждого эталона: фото → вектор ──► reference_images (PostgreSQL)
search    фото пользователя → вектор ──► ближайшие эталоны ──► группировка по вину ──► Top-K вин
          (HTTP POST /search у постоянно работающего сервиса или разовая CLI-команда)
evaluate  queries.csv ──► для каждого фото: как search ──► место правильного вина ──► Recall@K
```

Эталоны и фото пользователей проходят **один и тот же путь «фото → вектор»**
(`application/create_embedding.py`), поэтому их векторы сравнимы. Фото
пользователя никуда не сохраняется: для него только считается вектор.

### Шаг 1. Проверка файла (`LocalImageStorage`)

- путь приводится к абсолютному и должен остаться внутри `DATA_ROOT` —
  выход через `..` или символическую ссылку отклоняется;
- файл существует, расширение из `SUPPORTED_IMAGE_EXTENSIONS`
  (`.jpg`, `.jpeg`, `.png`, `.webp`), размер не больше `MAX_IMAGE_SIZE_BYTES`
  (10 МБ);
- файл полностью декодируется Pillow; если нет — `CorruptedImageError`.

### Шаг 2. Подготовка изображения (`ImagePreprocessor`)

- **поворот по EXIF**: телефоны хранят вертикальные снимки как
  горизонтальные пиксели плюс метку ориентации; без этого шага бутылка
  попала бы в модель лёжа;
- **прозрачность** (PNG с прозрачным фоном, частый формат каталожных
  фото) заливается белым;
- перевод в RGB.

Никакого кропа, выравнивания или улучшения здесь нет — это задача
`wine_label_preprocessing`.

### Шаг 3. Вектор DINOv2 (`DinoV2Embedder`)

1. Стандартный процессор модели: уменьшение до **256 px по короткой
   стороне** (бикубически), **центральный кроп 224×224**, нормализация
   средним и отклонением ImageNet.
2. Модель ViT-S/14 (12 слоёв) режет картинку на 16×16 = 256 патчей по
   14 px и добавляет служебный CLS-токен.
3. Берётся **CLS-токен последнего слоя — 384 числа** — описание всей
   картинки целиком.
4. Вектор **L2-нормализуется** (длина 1), поэтому косинусное сходство
   зависит только от «направления», а не от яркости или масштаба признаков.

Модель загружается один раз на запуск команды (в API — один раз на процесс),
веса кэшируются в `./model-cache`. Сейчас вычисления идут на CPU.

### Шаг 4. Хранение эталонов (`index`)

Эталоны берутся из `data/reference/`: `manifest.csv`, если он есть, иначе
фото в самой папке (одно фото — одно вино) и в подпапках (подпапка — одно
вино). Каждый эталон записывается в `reference_images`: `wine_id`, `slug`,
`image_uri`, `model_name`, `embedding VECTOR(384)`. Запись идёт как upsert по
`image_uri`: повторная индексация обновляет строку, а не создаёт дубликат.
Сначала проверяется весь manifest; битое фото попадает в `error_details`, но
не останавливает индексацию; ошибка базы — останавливает. С `--prune` из базы
удаляются эталоны, которых больше нет в источнике.

### Шаг 5. Поиск (`search`)

1. Вектор фото пользователя сравнивается со всеми эталонами той же модели
   по **косинусному расстоянию** pgvector:
   `ORDER BY embedding <=> query LIMIT n`. Расстояние 0 — одинаковые
   направления, 1 — не похожи, 2 — противоположны.
2. У вина может быть несколько эталонных фото, поэтому сначала берётся
   `n = max(RAW_RETRIEVAL_LIMIT, top_k)` ближайших **фотографий**, затем они
   группируются по `wine_id`, и каждое вино представлено своей ближайшей
   фотографией (`best_image_uri`).
3. Если в эти `n` фото попало меньше `top_k` разных вин, а в базе есть ещё
   фото, `n` удваивается и запрос повторяется, пока не наберётся `top_k` вин
   или не закончатся эталоны.
4. Вина сортируются по `score = 1 - distance` (1 — идентичный вектор; при
   одинаковом score — по `wine_id`), возвращаются первые `top_k`.

Векторного индекса (HNSW/IVFFlat) пока нет: каждый запрос — точный полный
перебор. Для десятков тысяч эталонов это миллисекунды; основное время
запроса — работа модели.

### Шаг 6. Оценка качества (`evaluate`)

Для каждой строки `queries.csv` выполняется шаг 5 на глубину
`max(20, top_k)` и находится место правильного `wine_id` среди разных вин.
`Recall@K` — доля успешно обработанных запросов, где правильное вино в
Top-K; считается для K = 1, 5, 20 и `--top-k`. Промахи перечисляются в
`incorrect_queries`, необработанные запросы — в `errors`.

### Что важно знать

- **Центральный кроп.** Процессор DINOv2 видит только центральный квадрат:
  у вертикального фото 3:4 теряется примерно по 17 % сверху и снизу, у
  9:16 — около четверти с каждой стороны. Лучше всего подавать фото,
  обрезанное по этикетке, — и эталоны, и запросы.
- **Одна модель на базу.** Колонка — `VECTOR(384)`; сравниваются только
  векторы с тем же `model_name`, а `image_uri` уникален, поэтому
  переиндексация другой моделью перезаписывает старые векторы.
- **Удаляет из базы только `index --prune`.** Без флага фото, которых
  больше нет в `data/reference`, остаются в поиске.
- **Вина одной серии визуально почти одинаковы.** На реальных фото
  «Жемчужная» Алиготе и «Жемчужная» Совиньон Блан (одинаковая этикетка,
  разный сорт) похожи друг на друга почти так же, как два снимка одной
  бутылки. DINOv2 отличает дизайн этикетки, а не мелкий текст на ней —
  для таких случаев нужен OCR (отдельный модуль).
- **После изменения подготовки изображений** (например, это обновление
  добавило поворот по EXIF и заливку прозрачности) эталоны нужно
  переиндексировать, чтобы они обрабатывались так же, как новые запросы.

## Архитектура

```text
entrypoints        CLI (argparse) и HTTP API (FastAPI)
    ↓
application        validate_image, create_embedding, index_reference_images,
    ↓              search_similar_wines, evaluate_retrieval, check_health
storage /          LocalImageStorage (проверка пути и файла),
preprocessing      ImagePreprocessor (Pillow → RGB)
    ↓
embedding          Embedder Protocol, DinoV2Embedder
    ↓
retrieval          Retriever, ReferenceRepository (интерфейсы)
    ↓
infrastructure     csv_manifest, reference_manifest, evaluation_manifest,
                   database: connection, PostgresReferenceRepository
    ↓
PostgreSQL + pgvector
```

Application-слой не обращается к psycopg напрямую: он работает через
интерфейсы из `retrieval/`, а реализацию на PostgreSQL подставляет CLI.

```text
src/dinov2_retrieval/
├── config.py                          # все настройки из переменных окружения
├── contracts.py                       # Pydantic-контракты входа и выхода
├── cli.py                             # совместимость с python -m ...cli
├── entrypoints/
│   ├── cli.py                         # validate/embed/index/search/evaluate/health
│   └── api.py                         # FastAPI + Swagger UI
├── application/
│   ├── validate_image.py
│   ├── create_embedding.py            # общий путь: проверка → RGB → embedding
│   ├── index_reference_images.py      # индексация эталонов
│   ├── search_similar_wines.py        # поиск и группировка по wine_id
│   ├── evaluate_retrieval.py          # Recall@K на размеченных запросах
│   └── check_health.py                # проверки для команды health
├── preprocessing/image_preprocessor.py
├── embedding/
│   ├── base.py                        # Embedder Protocol, EmbeddingError
│   └── dinov2_embedder.py             # DINOv2-small
├── retrieval/
│   ├── base.py                        # Retriever: search_similar
│   └── reference_repository.py        # ReferenceRepository, RepositoryError
└── infrastructure/
    ├── csv_manifest.py                # общее чтение CSV, проверка путей /data
    ├── reference_manifest.py          # manifest.csv и папки <slug>/
    ├── evaluation_manifest.py         # queries.csv
    ├── storage/local_storage.py       # LocalImageStorage
    └── database/
        ├── connection.py              # подключение, схема, инспекция
        ├── schema.sql
        └── postgres_reference_repository.py
database/init.sql                      # та же схема для первого старта PostgreSQL
examples/generate_demo_data.py         # синтетический демо-набор
```

`storage.py` в корне пакета — тонкий compatibility-фасад для старых импортов.

## Данные

Локальная папка `./data` монтируется в контейнер **только для чтения** как
`/data` (`DATA_ROOT`). Изображения вне `DATA_ROOT` отклоняются.

```text
data/
├── reference/                   # эталонные фотографии вин
│   ├── 87.6_20-08-2026.webp     # фото прямо в папке: одно фото = одно вино
│   ├── wine-001/                # подпапка: одно вино, несколько фото
│   │   ├── photo-1.jpg
│   │   └── photo-2.jpg
│   └── manifest.csv             # необязательно: свои wine_id и slug
├── evaluation/                  # размеченные фото пользователей для оценки
│   ├── queries.csv
│   └── images/
│       ├── query-001.jpg
│       └── query-002.jpg
├── queries/                     # фото пользователей для ручного поиска
│   └── test.jpeg
└── requests/                    # JSON-запросы поиска (например, от Airflow)
    └── search-request.json
```

Поддерживаются `.jpg`, `.jpeg`, `.png`, `.webp`, размер файла до
`MAX_IMAGE_SIZE_BYTES`.

**Пути внутри CSV** (`image_uri` в `manifest.csv`, `query_image_uri` в
`queries.csv`) — это пути **внутри контейнера**: абсолютные, начинаются с
`/data` и лежат внутри `DATA_ROOT`. Относительные пути, `/home/...` на хосте и
выход наружу через `..` отклоняются при чтении файла с номером строки.
Аргументы командной строки (`--manifest`, `--queries`, `--request-json`,
`--reference-dir`, `--image-uri`) можно указывать и относительно `DATA_ROOT`.

Папка `data/` не хранится в git.

## Реальный датасет

### 1. Эталонные фотографии

Эталон — фотография бутылки или этикетки известного вина. **Достаточно
положить фото в `data/reference/` и запустить `index` без аргументов** —
manifest не нужен:

```bash
docker compose run --rm dinov2-retrieval index
```

Как из файлов получаются вина:

| Где лежит фото | Вино (`wine_id` и `slug`) |
|---|---|
| `data/reference/87.6_20-08-2026.webp` | имя файла без расширения: `87.6_20-08-2026` — **одно фото = одно вино** |
| `data/reference/donum-2023/front.webp` | имя подпапки: `donum-2023` — все фото подпапки = одно вино |

Можно смешивать оба способа. Скрытые файлы, неподдерживаемые расширения и
вложенность глубже одной подпапки пропускаются. Подходят `.jpg`, `.jpeg`,
`.png`, `.webp` (в любом регистре) до 10 МБ; фото с телефона поворачиваются
по EXIF, прозрачный фон заливается белым, у анимированного WebP берётся
первый кадр.

**Если одно вино снято несколько раз, сложите эти фото в одну подпапку.**
Иначе каждое фото станет отдельным «вином», и в Top-K одно и то же вино
займёт несколько мест. Найти вероятные повторы можно запросом к базе после
индексации (пары с score > 0,85 — кандидаты, но это могут быть и разные вина
одной серии, проверьте глазами):

```bash
docker compose exec postgres psql -U dinov2 -d dinov2 -c "
SELECT round((1 - (a.embedding <=> b.embedding))::numeric, 3) AS score, a.wine_id, b.wine_id
FROM reference_images a JOIN reference_images b ON a.id < b.id
ORDER BY a.embedding <=> b.embedding LIMIT 20;"
```

На одно вино лучше 2–5 фотографий: разные ракурсы, освещение, фон. При поиске
вино представлено ближайшей из своих фотографий.

### 2. manifest.csv — свои идентификаторы из каталога

Если поиск должен возвращать ID вина из вашего каталога, а не имя файла или
папки, опишите фото в `data/reference/manifest.csv`. **Если этот файл есть,
`index` без аргументов берёт эталоны из него, а не из папки.**

CSV в UTF-8, разделитель — запятая:

```csv
wine_id,slug,image_uri
10231,chateau-margaux-2015,/data/reference/10231/front.webp
10231,chateau-margaux-2015,/data/reference/10231/angle.webp
10877,cloudy-bay-sauvignon-2022,/data/reference/10877/front.jpg
```

- `wine_id` — идентификатор вина из вашего каталога (его вернёт поиск);
- `slug` — человекочитаемое имя, у одного `wine_id` всегда одинаковое;
- `image_uri` — путь внутри контейнера, уникальный;
- лишние колонки (например, `vintage`, `source`) игнорируются, пустые строки
  пропускаются.

Если manifest не проходит проверку, индексация не начинается, а в ошибке
перечисляются номера проблемных строк (до 20 проблем за раз).

### 3. Фотографии для проверки качества

Это фотографии, **которых нет среди эталонов**, — такие, как их снимет
пользователь: телефон, свет магазина, фон, наклон. Положите их в
`data/evaluation/images/` и опишите в `data/evaluation/queries.csv`:

```csv
query_image_uri,wine_id
/data/evaluation/images/query-001.jpg,10231
/data/evaluation/images/query-002.jpg,10877
```

`wine_id` — **правильное** вино для этой фотографии, то есть тот же
`wine_id`, что в `manifest.csv`. `query_image_uri` не должен повторяться.
Если у `wine_id` нет ни одного проиндексированного эталона, запрос попадает в
`errors` (`WineNotIndexedError`) и в Recall не учитывается — это ошибка
разметки, а не модели.

Одна и та же фотография не должна быть и эталоном, и запросом: иначе поиск
найдёт саму себя, и Recall будет завышен.

### 4. Индексация, поиск, оценка

```bash
docker compose run --rm dinov2-retrieval index --prune
docker compose run --rm dinov2-retrieval search --image-uri /data/evaluation/images/query-001.jpg --top-k 20
docker compose run --rm dinov2-retrieval evaluate --queries /data/evaluation/queries.csv --top-k 20
```

### Замена демонстрационных данных

Синтетический демо-набор (в `reference/` и `evaluation/` есть файл
`DEMO.txt`) автоматически не удаляется. Чтобы перейти на реальные фото:

```bash
# 1. Убрать демо-файлы: перенести в data/demo/ (или удалить через rm -rf)
mkdir -p data/demo && mv data/reference data/evaluation data/demo/
mkdir -p data/reference data/evaluation/images

# 2. Положить реальные фото в data/reference/ (и, если нужно, manifest.csv)

# 3. Проиндексировать; --prune удалит из базы демо-эталоны,
#    которых больше нет в data/reference
docker compose run --rm dinov2-retrieval index --prune
```

Демо-набор можно создать заново (нужен `data/queries/test.jpeg`, существующие
`manifest.csv` и `queries.csv` без `--force` не перезаписываются):

```bash
docker run --rm -v "$PWD/data:/data" -v "$PWD/examples:/app/examples:ro" \
  --entrypoint python dinov2_retrieval-dinov2-retrieval \
  examples/generate_demo_data.py /data
```

(`docker run`, а не `docker compose run`: в compose `/data` смонтирован только
для чтения.)

### Почему демо-данные не показывают качество DINOv2

- 24 из 25 «вин» — нарисованные прямоугольники и круги разных цветов: они
  отличаются друг от друга гораздо сильнее, чем реальные этикетки одного
  производителя или соседних урожаев.
- Запросы получены из тех же картинок, что и эталоны (уменьшение, поворот,
  размытие), — это почти дубликаты, а не новые фотографии.
- `wine-001` — фото облаков, а не вина.
- Вин всего 25, поэтому Recall@20 почти ничего не проверяет: в Top-20 попадает
  80 % каталога.

Высокий Recall на демо-наборе означает только, что индексация, поиск и оценка
работают. Реальное качество показывает только `evaluate` на реальных
фотографиях пользователей.

## Конфигурация

Все настройки читаются из переменных окружения через `config.py`.

| Переменная                   | По умолчанию                    | Назначение                                    |
|------------------------------|---------------------------------|-----------------------------------------------|
| `DATA_ROOT`                  | обязательна                     | корень данных, в Docker `/data`               |
| `MAX_IMAGE_SIZE_BYTES`       | `10485760`                      | максимальный размер файла                     |
| `SUPPORTED_IMAGE_EXTENSIONS` | `.jpg,.jpeg,.png,.webp`         | разрешённые расширения (подмножество)         |
| `HF_HOME`                    | `/home/app/.cache/huggingface`  | кэш весов Hugging Face                        |
| `DATABASE_URL`               | обязательна для index/search/health | `postgresql://dinov2:dinov2@postgres:5432/dinov2` |
| `DINO_MODEL_NAME`            | `facebook/dinov2-small`         | модель                                        |
| `DINO_EMBEDDING_DIMENSION`   | `384`                           | ожидаемая размерность embedding               |
| `DEFAULT_TOP_K`              | `20`                            | сколько разных вин возвращает поиск           |
| `RAW_RETRIEVAL_LIMIT`        | `100`                           | сколько ближайших фото берётся до группировки (при нехватке разных вин удваивается) |

Схема БД создана под `VECTOR(384)`. Если сменить модель на другую размерность,
`health` покажет несовпадение с колонкой `embedding`.

## Запуск

```bash
cd modules/dinov2_retrieval
cp .env.example .env
mkdir -p data/queries data/reference data/evaluation/images data/requests model-cache
docker compose build
docker compose up -d postgres
```

Все зависимости (torch, transformers, psycopg, pgvector, pytest и т. д.)
ставятся только внутри Docker-образа. После изменения кода `docker compose
build` занимает секунды: зависимости лежат в отдельном слое, а скачанные
wheel-файлы кэшируются BuildKit.

`docker compose run` сам поднимает `postgres` и ждёт его готовности
(`depends_on: condition: service_healthy`).

### Кэш модели

`./model-cache` монтируется в `/home/app/.cache/huggingface` (`HF_HOME`).
Веса `facebook/dinov2-small` скачиваются при первом запуске `embed`, `index`,
`search` или `health` и дальше берутся из кэша.

## Команды CLI

Все команды печатают **один JSON-документ в stdout**, логи и ошибки дублируются
в stderr. В терминале JSON форматируется с отступами, без терминала (Airflow,
`docker compose run -T`, пайпы) печатается одной строкой.

### validate — проверить изображение

```bash
docker compose run --rm dinov2-retrieval validate --image-uri /data/queries/test.jpeg
```

```json
{"path": "/data/queries/test.jpeg", "width": 256, "height": 192, "mime_type": "image/jpeg"}
```

### embed — получить embedding

```bash
docker compose run --rm dinov2-retrieval embed --image-uri /data/queries/test.jpeg
```

```json
{"path": "/data/queries/test.jpeg", "model": "facebook/dinov2-small", "dimension": 384,
 "device": "cpu", "embedding_preview": [0.0593, 0.0727, 0.1058, -0.0046, 0.0652]}
```

Печатаются только первые 5 чисел; полный вектор возвращает
`application/create_embedding.py`.

### index — проиндексировать эталонные фотографии

```bash
docker compose run --rm dinov2-retrieval index                  # data/reference: manifest.csv, если есть, иначе фото в папке
docker compose run --rm dinov2-retrieval index --prune          # то же + удалить из базы фото, которых больше нет
docker compose run --rm dinov2-retrieval index --manifest /data/reference/manifest.csv
docker compose run --rm dinov2-retrieval index --reference-dir /data/reference
```

Откуда берутся эталоны и как фото превращаются в вина — см. «Реальный
датасет», пункты 1–2.

`--prune` синхронизирует базу с источником: удаляет эталоны той же модели,
которых нет в текущем списке (удалённые файлы, строки, убранные из manifest).
Фото, которые не удалось обработать в этом запуске, не удаляются; если не
обработано ни одно фото, удаление не выполняется вовсе — неверный путь не
очистит базу.

Модель загружается один раз на весь запуск; эталоны проходят ту же обработку,
что и фотография пользователя. Повторный запуск не создаёт дубликаты: запись с
тем же `image_uri` обновляется (`updated`). Ошибка одной фотографии не
останавливает индексацию и попадает в `error_details`.

```json
{
  "status": "partial",
  "model_name": "facebook/dinov2-small",
  "source": "folder /data/reference",
  "total": 3, "processed": 2, "inserted": 2, "updated": 0, "skipped": 0,
  "deleted": 0, "errors": 1,
  "error_details": [
    {"wine_id": "wine-002", "image_uri": "/data/reference/wine-002/photo-1.jpg",
     "error_type": "ImageNotFoundError",
     "message": "Image file does not exist: /data/reference/wine-002/photo-1.jpg"}
  ],
  "reference_count": 2
}
```

- `status`: `ok` — без ошибок, `partial` — часть фото не обработана,
  `failed` — не обработано ни одно фото.
- `source`: откуда взяты эталоны — `manifest <путь>` или `folder <путь>`.
- `skipped`: один и тот же файл, указанный в manifest дважды разными путями.
- `deleted`: сколько устаревших эталонов удалил `--prune` (без флага 0).
- `reference_count`: сколько эталонов этой модели сейчас в БД.

### search — найти похожие вина

```bash
docker compose run --rm dinov2-retrieval search --image-uri /data/queries/test.jpeg --top-k 20
docker compose run --rm dinov2-retrieval search --request-json /data/requests/search-request.json
```

Без `--top-k` используется `DEFAULT_TOP_K`, без `--request-id` генерируется
UUID. `--top-k`/`--request-id` нельзя совмещать с `--request-json`.

Алгоритм подробно описан в разделе «Как работает модуль», шаги 1–5.
Результат всегда содержит разные `wine_id`; меньше `top_k` вин он содержит,
только если в базе меньше вин.

Если в БД нет эталонов для модели, возвращается `"status": "no_results"`,
пустой `candidates` и `message` с подсказкой запустить `index`.

### evaluate — оценить качество поиска (Recall@K)

```bash
docker compose run --rm dinov2-retrieval evaluate --queries /data/evaluation/queries.csv --top-k 20
```

Для каждой строки `queries.csv` выполняется **ровно тот же путь, что у
`search`**: проверка файла → RGB → embedding той же модели → поиск в
PostgreSQL → группировка по `wine_id`. Затем проверяется, на каком месте среди
разных вин стоит правильный `wine_id`.

```text
Recall@K = число запросов, где правильный wine_id попал в Top-K
           / число успешно обработанных запросов
```

Всегда считаются Recall@1, Recall@5, Recall@20 и Recall для `--top-k` (по
умолчанию `DEFAULT_TOP_K`), поэтому поиск идёт на глубину `max(20, top_k)`.
Модель загружается один раз на весь запуск. Фото запросов в
`reference_images` не записываются. Ошибка одной фотографии (файла нет, не
декодируется, у вина нет эталонов) не останавливает оценку: запрос попадает в
`errors` и не входит в знаменатель. Ошибка базы данных останавливает оценку
(код 4).

```json
{
  "status": "ok",
  "model_name": "facebook/dinov2-small",
  "queries_total": 100,
  "queries_processed": 100,
  "queries_with_errors": 0,
  "top_k": 20,
  "recall_at_k": 0.78,
  "correct_queries": 78,
  "recall_at": {"1": 0.41, "5": 0.63, "20": 0.78},
  "correct_at": {"1": 41, "5": 63, "20": 78},
  "indexed_wines": 350,
  "incorrect_queries": [
    {
      "query_image_uri": "/data/evaluation/images/query-001.jpg",
      "expected_wine_id": "wine-001",
      "predicted_wine_ids": ["wine-003", "wine-005", "..."],
      "expected_rank": null,
      "reason": "expected wine is not in top-k"
    }
  ],
  "errors": [],
  "message": null
}
```

(Числа в примере условные.)

- `status`: `ok` — все запросы обработаны; `partial` — часть запросов с
  ошибками (код выхода 5); `failed` — не обработан ни один (код 1).
- `recall_at` / `correct_at` — Recall и число верных запросов для каждого K.
- `incorrect_queries` — обработанные запросы, где правильного вина нет в
  Top-`top_k`: `predicted_wine_ids` — что вернул поиск (до `top_k` вин),
  `expected_rank` — место правильного вина, если оно нашлось глубже
  `top_k` (но в пределах Top-20), иначе `null`.
- `errors` — запросы, которые не удалось обработать: `error_type` и `reason`.
- Embedding в отчёт не попадают.

Прогресс по каждому запросу (`hit`/`miss`, место) печатается в stderr.

### health — проверить окружение

```bash
docker compose run --rm dinov2-retrieval health
```

Проверяет конфигурацию, подключение к PostgreSQL, расширение `vector`, таблицу
`reference_images`, размерность колонки `embedding` и загрузку модели (с
пробным embedding). В БД ничего не меняется; пароль в `DATABASE_URL`
маскируется.

```json
{
  "status": "ok",
  "checks": {
    "config": {"status": "ok", "details": {"database_url": "postgresql://dinov2:***@postgres:5432/dinov2", "...": "..."}},
    "database": {"status": "ok", "details": {"server_version": "16.x"}},
    "pgvector": {"status": "ok", "details": {"version": "0.8.x"}},
    "reference_table": {"status": "ok", "details": {"embedding_dimension": 384, "reference_images": 50, "model_reference_images": 50}},
    "model": {"status": "ok", "details": {"model_name": "facebook/dinov2-small", "device": "cpu", "embedding_dimension": 384}}
  }
}
```

## Контракт для Apache Airflow

Запрос (`/data/requests/search-request.json`):

```json
{
  "request_id": "request-001",
  "image_uri": "/data/queries/test.jpeg",
  "top_k": 20
}
```

`request_id` и `image_uri` обязательны, `top_k` — целое больше нуля
(по умолчанию `DEFAULT_TOP_K`). Неизвестные поля отклоняются, чтобы опечатка
вроде `topk` не прошла незамеченной.

Ответ:

```json
{
  "request_id": "request-001",
  "status": "ok",
  "model_name": "facebook/dinov2-small",
  "query_embedding_dimension": 384,
  "candidates": [
    {
      "wine_id": "wine-001",
      "slug": "cabernet-2020",
      "score": 0.94,
      "distance": 0.06,
      "best_image_uri": "/data/reference/wine-001/photo-1.jpg"
    }
  ]
}
```

Ошибка любой команды:

```json
{"status": "error", "command": "search",
 "error": {"category": "input", "type": "ImageNotFoundError",
           "message": "Image file does not exist: /data/queries/missing.jpeg"}}
```

Коды завершения:

| Код | Категория  | Когда                                                                 |
|-----|------------|-----------------------------------------------------------------------|
| 0   | —          | успешно (в том числе `search` с `no_results`)                         |
| 1   | `input`    | файл/изображение, manifest, queries.csv, JSON-запрос, конфигурация; `index`/`evaluate` со статусом `failed` |
| 2   | `usage`    | неверные аргументы CLI                                                |
| 3   | `model`    | модель не загружается или вернула неверную размерность                |
| 4   | `database` | PostgreSQL недоступен, ошибка схемы или запроса                       |
| 5   | —          | `index`/`evaluate` завершены, но часть фотографий не обработана (`partial`) |

Без TTY результат — последняя строка stdout, поэтому его забирает XCom
(`BashOperator` и `DockerOperator` по умолчанию передают последнюю строку):

```python
BashOperator(
    task_id="search_wine",
    bash_command=(
        "cd /opt/svoevino/modules/dinov2_retrieval && "
        "docker compose run --rm -T dinov2-retrieval "
        "search --request-json /data/requests/{{ run_id }}.json"
    ),
)
```

## PostgreSQL

Сервис `postgres` (`pgvector/pgvector:pg16`) хранит данные в volume
`postgres_data` и доступен **только с этой машины** на `127.0.0.1:5433`
(порт меняется переменной `POSTGRES_PORT`; пароль `dinov2` — учебный, поэтому
порт не открыт в сеть). При первом старте
`database/init.sql` создаёт схему; команды `index` и `search` тоже создают её
при необходимости (`IF NOT EXISTS`).

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS reference_images (
    id BIGSERIAL PRIMARY KEY,
    wine_id TEXT NOT NULL,
    slug TEXT NOT NULL,
    image_uri TEXT NOT NULL UNIQUE,
    model_name TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

В таблице лежат только эталонные фотографии: embedding и метаданные
(`wine_id`, `slug`, `image_uri`, `model_name`). Сами изображения остаются в
`./data`, фотографии пользователей в БД не попадают.

Проверить содержимое:

```bash
docker compose exec postgres psql -U dinov2 -d dinov2 \
  -c "SELECT wine_id, slug, image_uri, vector_dims(embedding) FROM reference_images;"
```

`docker compose down` останавливает контейнеры и сохраняет данные;
`docker compose down -v` удаляет и volume с индексом.

## HTTP-сервис поиска (постоянно работающий)

Для приёма фото сервис запускается один раз и работает постоянно:

```bash
docker compose up -d          # PostgreSQL + сервис поиска на http://localhost:8000
docker compose ps             # дождаться "healthy" у dinov2-retrieval
docker compose logs -f dinov2-retrieval
```

**Модель DINOv2 загружается один раз при старте сервиса** и дальше
обрабатывает все запросы; соединение с PostgreSQL открывается на каждый
запрос. Контейнер перезапускается сам (`restart: unless-stopped`), пока его
не остановят `docker compose stop` / `down`. Сервис слушает только
`127.0.0.1` (порт меняется переменной `API_PORT`).

Сравнение со CLI: `docker compose run ... search` каждый раз запускает новый
контейнер и заново загружает модель (~6 с на запрос на CPU), поэтому CLI
подходит для разовых команд и Airflow, а фото от пользователей лучше
отправлять в сервис.

### Swagger: всё проверяется из браузера

Откройте <http://localhost:8000> — откроется Swagger UI. Кнопка «Try it out»
уже включена, под каждым ответом видно время запроса. Все операции, которые
есть в CLI, можно запускать отсюда:

| Раздел | Метод и путь | Что делает |
|---|---|---|
| Поиск | `POST /search` | **загрузить фото** (кнопка выбора файла) → Top-K разных вин |
| Поиск | `POST /search/uri` | то же для фото, которое уже лежит в `DATA_ROOT` |
| Эталоны | `POST /index` | проиндексировать `data/reference` (тело `{}`); `"prune": true` удалит пропавшие фото |
| Эталоны | `GET /references` | список загруженных вин и их фото (постранично) |
| Эталоны | `GET /images` | **показать фото** по пути, например `best_image_uri` из ответа поиска |
| Оценка качества | `POST /evaluate` | Recall@1/5/20 по `data/evaluation/queries.csv` (тело `{}`) |
| Служебное | `GET /health` | сервис запущен (для Docker healthcheck) |
| Служебное | `GET /health/details` | полная проверка, как CLI `health`; 503 при ошибке |
| Служебное | `POST /validate`, `POST /embed` | проверка файла и полный embedding |

Типичный сценарий проверки:

1. **Эталоны → `POST /index`** → Execute с примером по умолчанию
   `{"prune": true}` (другие примеры — в выпадающем списке «Examples»; поля
   `manifest` и `reference_dir` нужны, только если эталоны лежат не в
   `data/reference`, и указывать можно только одно из них). Ответ — та же
   статистика, что у CLI (`inserted`, `updated`, `errors`…). На CPU около
   0,4–0,6 с на фото; запрос ждёт окончания. Второй запуск индексации во
   время первого получит 409.
2. **Поиск → `POST /search`** → выбрать файл → Execute.
3. **Эталоны → `GET /images`** → вставить `best_image_uri` любого кандидата →
   Execute: Swagger покажет фото прямо в ответе. По умолчанию это уменьшенная
   до 800 px копия, повёрнутая по EXIF; `max_side=0` вернёт исходный файл.

Путь в `GET /images` (и во всех `image_uri`) — это путь **внутри
контейнера**, где `./data` видна как `/data`. Для файла
`./data/queries/test2.webp` подходят:

- `/data/queries/test2.webp` — абсолютный путь;
- `queries/test2.webp` — относительный путь считается от `/data`.

`data/queries/test2.webp` не подойдёт: он превратится в
`/data/data/queries/test2.webp`, и сервис вернёт 404 с подсказкой, где искал
файл. `GET /images` отдаёт только файлы поддерживаемых форматов внутри
`DATA_ROOT`: пути вне него получают 400.

### Поиск по загруженному фото

```bash
curl -s -X POST http://localhost:8000/search \
  -F "file=@/path/to/bottle.jpg" \
  -F "top_k=20"                       # необязательно, по умолчанию DEFAULT_TOP_K (20)
```

Необязательное поле `request_id` вернётся в ответе (по умолчанию — UUID).
Ответ такой же, как у CLI `search`:

```json
{
  "request_id": "5c0d…",
  "status": "ok",
  "model_name": "facebook/dinov2-small",
  "query_embedding_dimension": 384,
  "candidates": [
    {"wine_id": "96.42_22-08-2026_21-29-43", "slug": "96.42_22-08-2026_21-29-43",
     "score": 0.954, "distance": 0.046,
     "best_image_uri": "/data/reference/96.42_22-08-2026_21-29-43.webp"}
  ]
}
```

Загруженное фото проходит те же проверки и ту же подготовку, что эталоны:
формат jpg/jpeg/png/webp (по расширению имени файла, а без него — по
`Content-Type`), размер до `MAX_IMAGE_SIZE_BYTES`. Файл пишется во временную
папку контейнера только на время запроса и сразу удаляется — ни на диске, ни
в базе фото пользователя не остаётся.

### Поиск по фото из DATA_ROOT

```bash
curl -s -X POST http://localhost:8000/search/uri \
  -H 'Content-Type: application/json' \
  -d '{"image_uri": "/data/queries/test.jpeg", "top_k": 20}'
```

### Коды ответа

| Код | Когда |
|-----|-------|
| 200 | успех; `"status": "no_results"`, если эталоны ещё не проиндексированы |
| 400 | путь вне `DATA_ROOT` (`/search/uri`) |
| 404 | файла нет (`/search/uri`) |
| 413 | файл больше `MAX_IMAGE_SIZE_BYTES` |
| 415 | неподдерживаемый формат |
| 400 | ошибка в `manifest.csv` / `queries.csv` или файла нет (`/index`, `/evaluate`) |
| 409 | индексация уже идёт (`/index`) |
| 422 | файл не декодируется; нет поля `file`; неверный `top_k`; в `/index` указаны и `manifest`, и `reference_dir` |
| 503 | PostgreSQL недоступен или модель не загрузилась |

Индексация и оценка доступны и через CLI (`index`, `evaluate`), и через
сервис (`POST /index`, `POST /evaluate`); результат одинаковый. Новые эталоны
становятся доступны поиску сразу, перезапуск сервиса не нужен.

## Тесты

Unit-тесты не скачивают DINOv2, не требуют GPU и запущенного PostgreSQL
(модель и БД подменяются fake-объектами). Среди них: чтение и проверка
`manifest.csv`/`queries.csv`, Recall@1/5/20, найденное и не найденное вино,
битое изображение, группировка нескольких фото одного вина:

```bash
docker compose run --rm dinov2-retrieval pytest -q
```

Integration-тесты работают с настоящими PostgreSQL и моделью, по умолчанию
исключены и запускаются вручную. Тестовые строки удаляются после проверки:

```bash
docker compose up -d postgres
docker compose run --rm dinov2-retrieval pytest -m integration -q
```

Локально без Docker: `pip install -e '.[dev,db,api]'` и `pytest` — тесты,
которым нужны отсутствующие зависимости, пропускаются.
