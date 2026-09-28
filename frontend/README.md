# Frontend — сканер вина (Nuxt)

Одна страница, сделанная mobile-first: кнопка «Сфотографировать бутылку» → фото
уходит на сервер → страница опрашивает статус задачи (polling) → самое похожее
вино и сетка остальных кандидатов (до 20).

Nuxt 4 + Vue 3 + TypeScript. Пока настоящего асинхронного API нет, его
имитирует встроенный сервер Nuxt (Nitro) в `server/api/`.

## Запуск

Нужен Node.js `^22.19` или `^24.11`.

```bash
cd frontend
npm install            # если npm пишет ETARGET — npm install --prefer-online (устаревший кэш)
npm run dev            # http://localhost:3000, с горячей перезагрузкой
```

Production-сборка:

```bash
npm run build
node .output/server/index.mjs          # PORT=3100 HOST=0.0.0.0 — другой порт/доступ из сети
```

Проверка типов: `npm run typecheck`.

С телефона в той же сети: `npm run dev -- --host` и открыть
`http://<IP компьютера>:3000`. Камера открывается и по http-адресу — через
выбор файла, без доступа к `getUserMedia`.

## Демо

Встроенный мок отдаёт вымышленные вина (в шапке — плашка «Демо-данные»).
Одно и то же фото всегда даёт один и тот же результат.

| Параметр URL | Что делает |
|---|---|
| `?demo=1` | сразу запускает поиск на сгенерированном фото |
| `?scenario=slow` | обработка идёт ~13 с — видно смену этапов |
| `?scenario=error` | задача завершается с `state: "failed"` |
| `?scenario=empty` | ответ `no_results` — «Каталог пока пуст» |
| `?scenario=flaky` | каждый второй запрос статуса отвечает 503 — клиент это переживает |

## Контракт API (асинхронный, с polling)

Типы — [`shared/types/recognition.ts`](shared/types/recognition.ts).

**1. Загрузить фото** — `POST {apiBase}/search`, multipart: `file`, `top_k`.

```http
HTTP/1.1 202 Accepted
{"job_id": "f048fa3a…", "state": "queued", "poll_after_ms": 1000}
```

Ошибки: 413 — больше 10 МБ, 415 — не jpg/png/webp, 422 — файл не читается,
503 — сервис не готов.

**2. Опрашивать статус** — `GET {apiBase}/status/{job_id}`, ответ с
`Cache-Control: no-store`:

```json
{
  "job_id": "f048fa3a…",
  "state": "processing",
  "stage": "search",
  "progress": 0.59,
  "poll_after_ms": 1000,
  "result": null,
  "error": null
}
```

- `state`: `queued` → `processing` → `done` | `failed`;
- `stage`: `null` (в очереди), `prepare`, `search`, `rank` — показываются на экране поиска;
- при `done` в `result` лежит `SearchResponse` сервиса `modules/dinov2_retrieval`
  (`request_id, status, model_name, query_embedding_dimension, candidates[]`);
- при `failed` в `error` — `{code, message}`, `message` показывается пользователю;
- 404 — задача неизвестна или устарела.

**Как клиент опрашивает** ([`app/composables/useRecognition.ts`](app/composables/useRecognition.ts)):

- цепочка `setTimeout`, а не `setInterval`: следующий запрос уходит только после
  ответа на предыдущий, запросы не накладываются;
- пауза — `poll_after_ms` из ответа, иначе `pollIntervalMs` (1 с);
- сбой сети или 5xx — повтор с удвоением паузы (до 8 с); после
  `maxPollErrors` (3) сбоев подряд — ошибка;
- 404 — «Задача потерялась»; дольше `pollTimeoutMs` (90 с) — «слишком долго»;
- «Отменить», `Esc`, уход со страницы — `AbortController` прерывает и запрос,
  и опрос.

## Настройки

`runtimeConfig.public` в [`nuxt.config.ts`](nuxt.config.ts); каждое значение
переопределяется переменной окружения:

| Переменная | По умолчанию | Смысл |
|---|---|---|
| `NUXT_PUBLIC_API_BASE` | `/api` | адрес API; `/api` — встроенный мок |
| `NUXT_PUBLIC_DEMO_DATA` | `true` | плашка «Демо-данные»; `false` — брать фото эталонов из `GET {apiBase}/images` |
| `NUXT_PUBLIC_TOP_K` | `20` | сколько вин просить |
| `NUXT_PUBLIC_MAX_BYTES` | `10485760` | лимит размера фото |
| `NUXT_PUBLIC_POLL_INTERVAL_MS` | `1000` | пауза между опросами |
| `NUXT_PUBLIC_POLL_TIMEOUT_MS` | `90000` | сколько ждать результат |
| `NUXT_PUBLIC_MAX_POLL_ERRORS` | `3` | сбоев статуса подряд до ошибки |

## Подключение к настоящему бэкенду (отдельная задача)

Сейчас `modules/dinov2_retrieval` отдаёт результат синхронно (`POST /search`),
эндпоинтов задачи и статуса у него нет. Чтобы фронтенд работал с ним, бэкенду
нужно реализовать два эндпоинта из контракта выше — или поставить между ними
адаптер. Затем указать `NUXT_PUBLIC_API_BASE`, `NUXT_PUBLIC_DEMO_DATA=false`
и открыть CORS (или проксировать API через тот же origin).

## Mobile-first

- базовые стили — для телефона, крупнее — через `min-width` (480/600/720/820/980 px);
- на сенсорном экране главная кнопка открывает камеру (`capture="environment"`),
  рядом — «выбрать из галереи»; с мышью — обычный выбор файла, перетаскивание и вставка;
- в результатах на телефоне снизу закреплена панель «Сфотографировать другую»;
- учтены вырез экрана и полоса «домой» (`viewport-fit=cover`, `env(safe-area-inset-*)`);
- области нажатия от 44 px, hover-эффекты — только на устройствах с hover,
  `prefers-reduced-motion` отключает анимации.

## Структура

```text
app/
  pages/index.vue            сценарий страницы
  composables/useRecognition.ts   загрузка + polling статуса
  composables/useFileDrop.ts      перетаскивание и вставка
  components/                шапка, экран загрузки, прогресс, результат, карточки, сообщения
  utils/                     форматирование, проверка фото, палитры бутылок
  assets/css/main.css        дизайн-токены и mobile-first вёрстка
shared/
  types/recognition.ts       контракт API
  utils/hash.ts              хэш и детерминированный генератор
server/                      мок API (Nitro)
  api/search.post.ts         POST /api/search → 202
  api/status/[id].get.ts     GET  /api/status/:id
  utils/jobs.ts              очередь задач в памяти, этапы по времени
  utils/catalog.ts           демо-каталог
```
