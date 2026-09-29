import type { JobAccepted } from "#shared/types/recognition";
import { pipelineError } from "../utils/pipeline";
export default defineEventHandler(async (event): Promise<JobAccepted> => {
  const maxBytes = 10 * 1024 * 1024;
  if (Number(getRequestHeader(event, "content-length")) > maxBytes + 64 * 1024)
    throw createError({ statusCode: 413, message: "Фото больше 10 МБ" });
  const file = (await readMultipartFormData(event))?.find((part) => part.name === "file" && part.filename !== undefined);
  if (!file?.data.length) throw createError({ statusCode: 422, message: "Выберите изображение" });
  if (file.data.length > maxBytes) throw createError({ statusCode: 413, message: "Фото больше 10 МБ" });
  const form = new FormData();
  form.append("image", new Blob([new Uint8Array(file.data)], { type: file.type || "application/octet-stream" }), file.filename || "photo.jpg");
  try {
    const result = await $fetch<JobAccepted>(
      `${useRuntimeConfig(event).pipelineUrl}/v1/eval/predict?wait=false`,
      { method: "POST", body: form, retry: 0, timeout: 15_000 },
    );
    setResponseStatus(event, 202);
    setResponseHeader(event, "Cache-Control", "no-store");
    return result;
  } catch (error) { throw pipelineError(error); }
});
