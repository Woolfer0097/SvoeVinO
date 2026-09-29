export default defineEventHandler(async (event) => {
  setResponseHeader(event, "Cache-Control", "no-store");
  const body = await readBody(event);
  try {
    return await $fetch(`${useRuntimeConfig(event).pipelineUrl}/feedback`, {
      method: "POST", body, retry: 0, timeout: 90_000,
    });
  } catch (error) {
    const failure = error as { response?: { status?: number }; data?: { detail?: unknown } };
    const status = failure.response?.status ?? 503;
    const detail = failure.data?.detail;
    const jobMissing = status === 404 && typeof detail === "string" && detail.startsWith("Job not found or expired");
    let message = "Не удалось сохранить отзыв. Попробуйте ещё раз.";
    if (jobMissing) message = "Результат больше недоступен на сервере. Запустите распознавание ещё раз.";
    else if (status === 404) message = "Сервис отзывов недоступен. Попробуйте позже.";
    else if (status === 409) message = "Дождитесь завершения распознавания.";
    else if (status === 422) message = "Проверьте сведения о вине и год урожая.";
    throw createError({ statusCode: status === 404 && !jobMissing ? 503 : status, message });
  }
});
