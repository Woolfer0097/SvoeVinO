export function pipelineError(error: unknown) {
  const failed = error as { statusCode?: number; response?: { status?: number } };
  const code = failed.statusCode ?? failed.response?.status;
  return createError({
    statusCode: code && code >= 400 && code < 600 ? code : 503,
    message: code === 404 ? "Задача не найдена" : "Сервис распознавания недоступен",
  });
}
