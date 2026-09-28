// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  compatibilityDate: "2026-09-28",
  devtools: { enabled: false },
  css: ["~/assets/css/main.css"],

  app: {
    head: {
      htmlAttrs: { lang: "ru" },
      title: "SvoeVinO — сканер российских вин",
      // viewport-fit=cover — чтобы учитывать вырез и полосу «домой» через env(safe-area-inset-*).
      viewport: "width=device-width, initial-scale=1, viewport-fit=cover",
      meta: [
        { name: "description", content: "Прототип: загрузите фото бутылки и посмотрите самые похожие вина из каталога российских вин." },
        { name: "theme-color", content: "#FBF8F4" },
      ],
      link: [
        {
          rel: "icon",
          href: "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='8' fill='%237B1E3C'/%3E%3Cpath d='M11 6h10l-.8 6.5A4.3 4.3 0 0 1 16 16.3a4.3 4.3 0 0 1-4.2-3.8z' fill='%23F3DDB0'/%3E%3Cpath d='M16 16.3V25M12.5 25h7' stroke='%23F3DDB0' stroke-width='1.8' stroke-linecap='round'/%3E%3C/svg%3E",
        },
        { rel: "preconnect", href: "https://fonts.googleapis.com" },
        { rel: "preconnect", href: "https://fonts.gstatic.com", crossorigin: "" },
        {
          rel: "stylesheet",
          href: "https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,500;0,600;0,700;1,600&family=Manrope:wght@400;500;600;700&display=swap",
        },
      ],
    },
  },

  // Любое значение переопределяется переменной окружения NUXT_PUBLIC_*,
  // например NUXT_PUBLIC_API_BASE=http://localhost:8000.
  runtimeConfig: {
    public: {
      // Базовый адрес API распознавания. "/api" — встроенный мок из server/api.
      apiBase: "/api",
      // Показывать плашку «Демо-данные» (мок отдаёт вымышленные вина).
      demoData: true,
      // Сколько разных вин просить у поиска (как DEFAULT_TOP_K в бэкенде).
      topK: 20,
      // Максимальный размер фото (как MAX_IMAGE_SIZE_BYTES в бэкенде, 10 МиБ).
      maxBytes: 10 * 1024 * 1024,
      // Polling: пауза между запросами статуса, если сервер не прислал poll_after_ms.
      pollIntervalMs: 1000,
      // Сколько ждать результата, прежде чем показать ошибку.
      pollTimeoutMs: 90_000,
      // Сколько подряд неудачных запросов статуса терпим до ошибки.
      maxPollErrors: 3,
    },
  },
});
