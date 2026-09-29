import type { JobStatus } from "#shared/types/recognition";
import { pipelineError } from "../../utils/pipeline";
export default defineEventHandler(async (event): Promise<JobStatus> => {
  setResponseHeader(event, "Cache-Control", "no-store");
  const id = getRouterParam(event, "id") ?? "";
  if (!/^[a-f0-9]{32}$/.test(id)) throw createError({ statusCode: 404, message: "Задача не найдена" });
  try {
    return await $fetch<JobStatus>(`${useRuntimeConfig(event).pipelineUrl}/image/status`, {
      query: { job_id: id }, retry: 0, timeout: 10_000,
    });
  } catch (error) { throw pipelineError(error); }
});
