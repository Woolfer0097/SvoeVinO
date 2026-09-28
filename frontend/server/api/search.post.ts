/*
 * Мок: принять фото и поставить задачу распознавания в очередь.
 * Проверки повторяют бэкенд (jpg/png/webp, до 10 МиБ). Само фото не
 * сохраняется: для детерминированного ответа берутся только имя и размер.
 * ?scenario=slow|error|empty|flaky — демо-сценарии.
 */
import type { JobAccepted } from "#shared/types/recognition";

const MAX_BYTES = 10 * 1024 * 1024;
const TYPES_BY_EXTENSION: Record<string, string> = {
  jpg: "image/jpeg",
  jpeg: "image/jpeg",
  png: "image/png",
  webp: "image/webp",
};
const ALLOWED_TYPES = new Set(Object.values(TYPES_BY_EXTENSION));

function reject(statusCode: number, detail: string): never {
  throw createError({ statusCode, data: { detail } });
}

export default defineEventHandler(async (event): Promise<JobAccepted> => {
  const parts = await readMultipartFormData(event);
  const file = parts?.find((part) => part.name === "file" && part.filename !== undefined);
  if (!file) reject(422, "Field 'file' is required");
  if (file.data.length === 0) reject(422, "Uploaded image is empty");
  if (file.data.length > MAX_BYTES) reject(413, `Image is too large; maximum is ${MAX_BYTES} bytes`);

  const extension = file.filename?.split(".").pop()?.toLowerCase() ?? "";
  const type = file.type?.toLowerCase() || TYPES_BY_EXTENSION[extension] || "";
  if (!ALLOWED_TYPES.has(type)) reject(415, "Unsupported image format. Supported: jpg, jpeg, png, webp");

  const topKPart = parts?.find((part) => part.name === "top_k");
  const topK = Math.min(50, Math.max(1, Number(topKPart?.data.toString()) || 20));

  const accepted = createJob(parseScenario(getQuery(event).scenario), `${file.filename}|${file.data.length}`, topK);
  setResponseStatus(event, 202);
  return accepted;
});
