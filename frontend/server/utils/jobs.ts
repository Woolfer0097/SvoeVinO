/*
 * Очередь задач мок-сервера в памяти процесса. Статус вычисляется по времени,
 * прошедшему с создания задачи, — как будто её обрабатывает воркер.
 */
import type { JobAccepted, JobStage, JobStatus } from "#shared/types/recognition";
import { buildEmptyResult, buildMockResult } from "./catalog";

export const SCENARIOS = ["normal", "slow", "error", "empty", "flaky"] as const;
export type Scenario = (typeof SCENARIOS)[number];

interface MockJob {
  id: string;
  createdAt: number;
  scenario: Scenario;
  seed: string;
  topK: number;
  statusCalls: number;
}

const POLL_AFTER_MS = 1000;
const JOB_TTL_MS = 10 * 60_000;
const FAIL_AT_MS = 1800;

// Когда заканчивается каждый этап (мс от создания задачи) в обычном сценарии.
const TIMELINE: { until: number; stage: JobStage | null }[] = [
  { until: 500, stage: null },
  { until: 1300, stage: "prepare" },
  { until: 2500, stage: "search" },
  { until: 3200, stage: "rank" },
];

const jobs = new Map<string, MockJob>();

export function parseScenario(value: unknown): Scenario {
  return SCENARIOS.includes(value as Scenario) ? (value as Scenario) : "normal";
}

export function createJob(scenario: Scenario, seed: string, topK: number): JobAccepted {
  const now = Date.now();
  for (const [id, job] of jobs) {
    if (now - job.createdAt > JOB_TTL_MS) jobs.delete(id);
  }

  const id = crypto.randomUUID().replaceAll("-", "");
  jobs.set(id, { id, createdAt: now, scenario, seed, topK, statusCalls: 0 });
  return { job_id: id, state: "queued", poll_after_ms: POLL_AFTER_MS };
}

export function findJob(id: string): MockJob | undefined {
  return jobs.get(id);
}

/** Сценарий flaky: каждый второй запрос статуса отвечает 503. */
export function isFlakyFailure(job: MockJob): boolean {
  job.statusCalls += 1;
  return job.scenario === "flaky" && job.statusCalls % 2 === 0;
}

export function statusOf(job: MockJob): JobStatus {
  const scale = job.scenario === "slow" ? 4 : 1;
  const elapsed = (Date.now() - job.createdAt) / scale;
  const total = TIMELINE[TIMELINE.length - 1]!.until;
  const base = { job_id: job.id, poll_after_ms: POLL_AFTER_MS, result: null, error: null };

  if (job.scenario === "error" && elapsed >= FAIL_AT_MS) {
    return {
      ...base,
      state: "failed",
      stage: "search",
      progress: FAIL_AT_MS / total,
      error: { code: "search_unavailable", message: "Поиск временно недоступен: база эталонов не отвечает." },
    };
  }

  const current = TIMELINE.find((step) => elapsed < step.until);
  if (current) {
    return {
      ...base,
      state: current.stage ? "processing" : "queued",
      stage: current.stage,
      progress: Math.min(0.99, elapsed / total),
    };
  }

  return {
    ...base,
    state: "done",
    stage: "rank",
    progress: 1,
    result: job.scenario === "empty" ? buildEmptyResult(job.id) : buildMockResult(job.id, job.seed, job.topK),
  };
}
