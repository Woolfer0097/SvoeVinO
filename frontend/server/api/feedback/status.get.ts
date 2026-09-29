import { pipelineError } from "../../utils/pipeline";
export default defineEventHandler(async (event) => {
  setResponseHeader(event, "Cache-Control", "no-store");
  const id = getQuery(event).job_id;
  if (typeof id !== "string" || !/^[a-f0-9]{32}$/.test(id))
    throw createError({ statusCode: 422, message: "Некорректный номер отзыва" });
  try {
    return await $fetch(`${useRuntimeConfig(event).pipelineUrl}/feedback/status`, {
      query: { job_id: id }, retry: 0, timeout: 10_000,
    });
  } catch (error) { throw pipelineError(error); }
});
