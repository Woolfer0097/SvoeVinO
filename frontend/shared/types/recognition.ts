/*
 * Контракт асинхронного распознавания: фото ставится в очередь, клиент
 * периодически опрашивает статус (polling), пока задача не завершится.
 *
 *   POST {apiBase}/search            multipart: file, top_k   → 202 JobAccepted
 *   GET  {apiBase}/status/{job_id}                             → 200 JobStatus
 *                                                                404 — задача не найдена или устарела
 *
 * Результат готовой задачи — SearchResponse из
 * modules/dinov2_retrieval/src/dinov2_retrieval/contracts.py.
 */

/** Кандидат поиска (WineCandidate в dinov2_retrieval). */
export interface WineCandidate {
  wine_id: string;
  slug: string;
  score: number;
  distance: number;
  best_image_uri: string;
  // Поля карточки вина — план: бэкенд их пока не возвращает.
  name?: string;
  winery?: string;
  region?: string;
  grape_variety?: string;
  color?: string;
}

/** Top-K разных вин для одного фото (SearchResponse в dinov2_retrieval). */
export interface SearchResponse {
  request_id: string;
  status: "ok" | "no_results";
  model_name: string;
  query_embedding_dimension: number;
  candidates: WineCandidate[];
  message?: string | null;
}

export type JobState = "queued" | "processing" | "done" | "failed";

/** Этап обработки на сервере; null — задача ещё в очереди. */
export type JobStage = "prepare" | "search" | "rank";

/** Ответ на загрузку фото: задача принята (HTTP 202). */
export interface JobAccepted {
  job_id: string;
  state: JobState;
  /** Через сколько миллисекунд имеет смысл спросить статус. */
  poll_after_ms?: number;
}

export interface JobError {
  code: string;
  message: string;
}

/** Ответ эндпоинта статуса. */
export interface JobStatus {
  job_id: string;
  state: JobState;
  stage: JobStage | null;
  /** Общий прогресс задачи от 0 до 1. */
  progress: number;
  poll_after_ms?: number;
  /** Заполнен, когда state = "done". */
  result: SearchResponse | null;
  /** Заполнен, когда state = "failed". */
  error: JobError | null;
}
