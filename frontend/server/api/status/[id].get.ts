/*
 * Мок: статус задачи распознавания. Клиент опрашивает его периодически,
 * пока state не станет "done" или "failed".
 */
import type { JobStatus } from "#shared/types/recognition";

export default defineEventHandler((event): JobStatus => {
  setResponseHeader(event, "Cache-Control", "no-store");

  const job = findJob(getRouterParam(event, "id") ?? "");
  if (!job) {
    throw createError({ statusCode: 404, data: { detail: "Job not found or expired" } });
  }
  if (isFlakyFailure(job)) {
    throw createError({ statusCode: 503, data: { detail: "Status is temporarily unavailable" } });
  }
  return statusOf(job);
});
