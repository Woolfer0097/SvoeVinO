/*
 * Распознавание фото: загрузка → задача в очереди → опрос статуса (polling)
 * до "done"/"failed". Контракт — shared/types/recognition.ts.
 */
import type { JobAccepted, JobStage, JobStatus, SearchResponse } from "#shared/types/recognition";

export type RecognitionPhase = "idle" | "uploading" | "polling" | "done" | "empty" | "error";

export interface RecognitionProblem {
  kind: ProblemKind;
  title: string;
  message: string;
}

type ProblemKind =
  | "network"
  | "too_large"
  | "unsupported"
  | "unreadable"
  | "unavailable"
  | "timeout"
  | "job_lost"
  | "job_failed"
  | "unknown";

const PROBLEMS: Record<ProblemKind, [string, string]> = {
  network: ["Сервис недоступен", "Не удалось связаться с сервером. Проверьте подключение и попробуйте ещё раз."],
  too_large: ["Фото слишком большое", "Сервер принимает файлы до 10 МБ. Уменьшите фото и попробуйте снова."],
  unsupported: ["Формат не подходит", "Загрузите фото в формате JPG, PNG или WebP."],
  unreadable: ["Не получилось прочитать фото", "Файл повреждён или это не изображение. Попробуйте другое фото."],
  unavailable: ["Сервис временно недоступен", "Модель или база данных ещё не готовы. Попробуйте через минуту."],
  timeout: ["Поиск занял слишком много времени", "Задача может ещё выполняться. Нажмите «Проверить результат», чтобы продолжить ожидание без повторной отправки фото."],
  job_lost: ["Задача потерялась", "Сервер больше не знает об этом поиске — возможно, он перезапустился. Запустите поиск заново."],
  job_failed: ["Не удалось найти вино", "Во время поиска произошла ошибка. Попробуйте ещё раз."],
  unknown: ["Что-то пошло не так", "Сервер ответил ошибкой. Попробуйте ещё раз."],
};

const MAX_BACKOFF_MS = 8000;

class RecognitionFailure extends Error {
  constructor(
    readonly kind: ProblemKind,
    readonly serverMessage?: string,
  ) {
    super(kind);
  }
}

function problemOf(kind: ProblemKind, serverMessage?: string): RecognitionProblem {
  const [title, message] = PROBLEMS[kind];
  return { kind, title, message: serverMessage || message };
}

function statusCodeOf(error: unknown): number | undefined {
  const candidate = error as { statusCode?: number; status?: number; response?: { status?: number } };
  return candidate?.statusCode ?? candidate?.status ?? candidate?.response?.status;
}

function kindByUploadStatus(status: number | undefined): ProblemKind {
  if (status === undefined) return "network";
  if (status === 413) return "too_large";
  if (status === 415) return "unsupported";
  if (status === 422) return "unreadable";
  if (status === 503) return "unavailable";
  return "unknown";
}

function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(signal.reason);
      return;
    }
    const aborted = () => {
      clearTimeout(timer);
      reject(signal.reason);
    };
    const timer = setTimeout(() => {
      signal.removeEventListener("abort", aborted);
      resolve();
    }, ms);
    signal.addEventListener("abort", aborted, { once: true });
  });
}

export function useRecognition() {
  const config = useRuntimeConfig().public;

  const phase = ref<RecognitionPhase>("idle");
  const stage = ref<JobStage | null>(null);
  const progress = ref(0);
  const jobId = ref<string | null>(null);
  const result = shallowRef<SearchResponse | null>(null);
  const problem = ref<RecognitionProblem | null>(null);
  const lastFile = shallowRef<File | null>(null);
  let controller: AbortController | null = null;

  const isBusy = computed(() => phase.value === "uploading" || phase.value === "polling");
  const canResume = computed(() => jobId.value !== null
    && (problem.value?.kind === "timeout" || problem.value?.kind === "network"));

  async function upload(file: File, scenario: string | undefined, signal: AbortSignal): Promise<JobAccepted> {
    const form = new FormData();
    form.append("file", file, file.name || "photo.jpg");
    try {
      return await $fetch<JobAccepted>(`${config.apiBase}/search`, {
        method: "POST",
        body: form,
        query: scenario ? { scenario } : undefined,
        signal,
      });
    } catch (error) {
      if (signal.aborted) throw error;
      throw new RecognitionFailure(kindByUploadStatus(statusCodeOf(error)));
    }
  }

  async function poll(accepted: JobAccepted, signal: AbortSignal): Promise<JobStatus> {
    const startedAt = Date.now();
    let delay = accepted.poll_after_ms ?? config.pollIntervalMs;
    let failures = 0;

    for (;;) {
      await sleep(delay, signal);
      if (Date.now() - startedAt > config.pollTimeoutMs) throw new RecognitionFailure("timeout");

      let status: JobStatus;
      try {
        status = await $fetch<JobStatus>(`${config.apiBase}/status/${encodeURIComponent(accepted.job_id)}`, {
          signal,
          retry: 0,
        });
      } catch (error) {
        if (signal.aborted) throw error;
        if (statusCodeOf(error) === 404) throw new RecognitionFailure("job_lost");
        // Сеть мигнула или сервер перегружен: ждём дольше и пробуем снова.
        failures += 1;
        if (failures > config.maxPollErrors) throw new RecognitionFailure("network");
        delay = Math.min(delay * 2, MAX_BACKOFF_MS);
        continue;
      }

      failures = 0;
      stage.value = status.stage;
      progress.value = Math.max(progress.value, Math.min(1, Math.max(0, status.progress)));
      if (status.state === "done") return status;
      if (status.state === "failed") throw new RecognitionFailure("job_failed", status.error?.message);
      delay = status.poll_after_ms ?? config.pollIntervalMs;
    }
  }

  async function start(file: File, options: { scenario?: string } = {}) {
    cancel();
    const current = new AbortController();
    controller = current;
    const { signal } = current;

    lastFile.value = file;
    phase.value = "uploading";
    stage.value = null;
    progress.value = 0;
    jobId.value = null;
    result.value = null;
    problem.value = null;

    try {
      const accepted = await upload(file, options.scenario, signal);
      jobId.value = accepted.job_id;
      phase.value = "polling";

      await awaitResult(accepted, signal);
    } catch (error) {
      if (signal.aborted) return;
      problem.value =
        error instanceof RecognitionFailure ? problemOf(error.kind, error.serverMessage) : problemOf("unknown");
      phase.value = "error";
    } finally {
      if (controller === current) controller = null;
    }
  }

  async function awaitResult(accepted: JobAccepted, signal: AbortSignal) {
    const status = await poll(accepted, signal);
    const response = status.result;
    result.value = response;
    // Weak matches still have useful candidates; do not hide them.
    phase.value = response?.candidates.length ? "done" : "empty";
  }

  async function resume() {
    const existingId = jobId.value;
    if (!existingId || isBusy.value) return;
    cancel();
    const current = new AbortController();
    controller = current;
    phase.value = "polling";
    problem.value = null;
    try {
      await awaitResult({ job_id: existingId, state: "processing", poll_after_ms: 0 }, current.signal);
    } catch (error) {
      if (current.signal.aborted) return;
      problem.value = error instanceof RecognitionFailure
        ? problemOf(error.kind, error.serverMessage) : problemOf("unknown");
      phase.value = "error";
    } finally {
      if (controller === current) controller = null;
    }
  }

  function retry(options: { scenario?: string } = {}) {
    if (isBusy.value) return;
    if (canResume.value) return resume();
    if (lastFile.value) return start(lastFile.value, options);
  }

  /** Прекратить опрос; задача на сервере доживёт до своего TTL. */
  function cancel() {
    controller?.abort();
    controller = null;
    if (isBusy.value) phase.value = "idle";
  }

  function reset() {
    cancel();
    phase.value = "idle";
    result.value = null;
    problem.value = null;
    jobId.value = null;
    lastFile.value = null;
    stage.value = null;
    progress.value = 0;
  }

  onScopeDispose(cancel);

  return {
    phase: readonly(phase),
    stage: readonly(stage),
    progress: readonly(progress),
    jobId: readonly(jobId),
    result,
    problem: readonly(problem),
    isBusy,
    canResume,
    start,
    retry,
    cancel,
    reset,
  };
}
