import { expect, test } from "@playwright/test";

test.beforeEach(async ({ page }) => {
  page.on("pageerror", error => { console.error("Browser error:", error.message); });
});

test("minimal upload screen fits the viewport", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator('[data-ready="true"]')).toBeVisible({ timeout: 60_000 });
  await expect(page.getByRole("heading", { name: "Что за вино?" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Сделать фото", exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Выбрать из галереи" })).toBeVisible();
  await expect(page.getByText("Демо-данные")).toHaveCount(0);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

test("camera denial is recoverable without permission changes", async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(navigator.mediaDevices, "getUserMedia", {
      value: async () => { throw new DOMException("Denied", "NotAllowedError"); },
    });
  });
  await page.goto("/");
  await expect(page.locator('[data-ready="true"]')).toBeVisible({ timeout: 60_000 });
  await page.getByRole("button", { name: "Сделать фото", exact: true }).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await expect(page.getByText("Камера недоступна.", { exact: false })).toBeVisible();
  await page.getByRole("button", { name: "Закрыть камеру" }).click();
  await expect(page.getByRole("dialog")).toHaveCount(0);
});

test("invalid photo shows an actionable error", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator('[data-ready="true"]')).toBeVisible({ timeout: 60_000 });
  await page.locator('input[type="file"]:not([capture])').setInputFiles({
    name: "broken.png", mimeType: "image/png", buffer: Buffer.from("not an image"),
  });
  await expect(page.getByRole("heading", { name: "Не получилось прочитать фото" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Выбрать другое фото" })).toBeVisible();
});

test("real catalog result shows one complete card, transparent photo and link", async ({ page, request }, testInfo) => {
  const uri = "/data/web_photos/avtohtonnoe-vino-kryma-beloe-suhoe-2dbb737ff6.webp";
  const image = await request.get("/api/images", { params: { uri } });
  expect(image.status()).toBe(200);
  expect(image.headers()["content-type"]).toContain("image/webp");
  await page.goto("/");
  await expect(page.locator('[data-ready="true"]')).toBeVisible({ timeout: 60_000 });
  await page.locator('input[type="file"]:not([capture])').setInputFiles({
    name: "team-catalog.webp", mimeType: "image/webp", buffer: await image.body(),
  });
  await expect(page.getByRole("heading", { name: "Автохтонное Вино Крыма белое сухое", exact: true })).toBeVisible({ timeout: 110_000 });
  const link = page.getByRole("link", { name: "Открыть на Вино своё" });
  await expect(link).toHaveAttribute("href", "https://vino-svoe.ru/wines/avtohtonnoe-vino-kryma-beloe-suhoe");
  await expect(page.getByText("Валерий Захарьин", { exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "О вине", exact: true })).toBeVisible();
  await expect(page.getByText("В аромате присутствует фруктово-цветочное направление", { exact: false })).toBeVisible();
  await expect(page.locator(".similar-grid")).toHaveCount(0);
  await expect(page.getByText("сходства", { exact: false })).toHaveCount(0);
  const wineImage = page.locator(".result-card__photo img");
  await expect(wineImage).toBeVisible();
  await expect.poll(() => wineImage.evaluate((image) => (image as HTMLImageElement).naturalWidth)).toBeGreaterThan(0);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: testInfo.outputPath("result.png"), fullPage: true });
  const queryToggle = page.getByRole("button", { name: "Ваш снимок", exact: true });
  await expect(queryToggle).toHaveAttribute("aria-expanded", "false");
  await queryToggle.click();
  const expandedToggle = page.getByRole("button", { name: "Скрыть снимок", exact: true });
  await expect(expandedToggle).toHaveAttribute("aria-expanded", "true");
  const queryPhoto = page.getByRole("img", { name: "Ваш снимок этикетки для сравнения", exact: true });
  await expect(queryPhoto).toBeVisible();
  const sourceFrame = page.locator(".query-photo__frame");
  await expect.poll(async () => {
    const source = await sourceFrame.boundingBox();
    const reference = await page.locator(".result-card__photo").boundingBox();
    return !!source && !!reference && Math.abs(source.y - reference.y) < 2
      && source.x + source.width <= reference.x + 2 && source.width > 100;
  }).toBe(true);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: testInfo.outputPath("comparison.png"), fullPage: true });
  await expandedToggle.focus();
  await page.keyboard.press("Space");
  await expect(page.locator(".comparison-query")).not.toBeVisible();
  // The catalog photo also opens comparison; animations respect accessibility.
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.getByRole("button", { name: "Сравнить ваш снимок с фото из каталога", exact: true }).click();
  await expect(queryPhoto).toBeVisible();
  expect(await page.locator(".result-card__images").evaluate(el => getComputedStyle(el).transitionDuration)).toBe("0s");
  await page.getByRole("button", { name: "Скрыть сравнение снимков", exact: true }).click();
  await expect(page.locator(".comparison-query")).not.toBeVisible();
});


test("loading copy follows server stages, including degraded branches", async ({ page }) => {
  const stages = [
    ["prepare", "Проверяем снимок"], ["preprocess", "Готовим этикетку"],
    ["preprocess_done", "Фото готово к поиску"], ["search", "Запускаем распознавание"],
    ["dino_started", "Ищем похожие этикетки"], ["dino_done", "Похожие этикетки найдены"],
    ["ocr_started", "Читаем название и год"], ["ocr_done", "Текст этикетки прочитан"],
    ["superpoint_started", "Сверяем детали этикетки"], ["superpoint_done", "Визуальная проверка завершена"],
    ["ocr_rescue_started", "Перепроверяем находки по тексту"], ["ocr_rescue_done", "Дополнительная проверка завершена"],
    ["fusion", "Объединяем фото и текст"], ["color", "Сравниваем цвета этикеток"],
    ["vintage", "Сверяем год урожая"], ["ocr_failed", "Продолжаем по фотографии"],
    ["superpoint_failed", "Уточняем по другим признакам"], ["ocr_rescue_failed", "Продолжаем с проверенными винами"],
    ["future_stage", "Продолжаем распознавание"],
  ];
  let stage: string | null = null;
  await page.route("**/api/search", route => route.fulfill({
    status: 202, json: { job_id: "a".repeat(32), state: "queued", poll_after_ms: 80 },
  }));
  // Contract fixture only: production still displays real backend stages.
  await page.route("**/api/status/*", route => route.fulfill({ json: {
    job_id: "a".repeat(32), state: "processing", stage, progress: .6,
    poll_after_ms: 80, result: null, error: null,
  } }));
  await page.goto("/");
  await expect(page.locator('[data-ready="true"]')).toBeVisible({ timeout: 60_000 });
  await page.locator('input[type="file"]:not([capture])').setInputFiles({
    name: "stages.png", mimeType: "image/png",
    buffer: Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=", "base64"),
  });
  await expect(page.getByRole("heading", { name: "Ваше фото в очереди…", exact: true })).toBeVisible();
  for (const [nextStage, heading] of stages) {
    stage = nextStage!;
    await expect(page.getByRole("heading", { name: `${heading}…`, exact: true })).toBeVisible();
  }
  await page.getByRole("button", { name: "Отменить", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Что за вино?" })).toBeVisible();
});
