import type { JobStage } from "#shared/types/recognition";
import type { RecognitionPhase } from "~/composables/useRecognition";

export interface ProgressCopy { heading: string; detail: string }

export const STAGE_COPY: Record<JobStage, ProgressCopy> = {
  prepare: { heading: "Проверяем снимок", detail: "Проверяем изображение перед распознаванием." },
  preprocess: { heading: "Готовим этикетку", detail: "Настраиваем размер снимка и улучшаем читаемость текста." },
  preprocess_done: { heading: "Фото готово к поиску", detail: "Передаём подготовленный снимок на сравнение с каталогом." },
  search: { heading: "Запускаем распознавание", detail: "Ищем визуальные совпадения и читаем текст этикетки." },
  dino_started: { heading: "Ищем похожие этикетки", detail: "Сравниваем оформление бутылки с фотографиями каталога." },
  dino_done: { heading: "Похожие этикетки найдены", detail: "Переходим к проверке мелких деталей на фотографиях." },
  ocr_started: { heading: "Читаем название и год", detail: "Распознаём надписи, сорт винограда и год на этикетке." },
  ocr_done: { heading: "Текст этикетки прочитан", detail: "Продолжаем проверку совпадений по деталям фотографии." },
  ocr_failed: { heading: "Продолжаем по фотографии", detail: "Текст прочитать не удалось. Используем визуальные совпадения." },
  superpoint_started: { heading: "Сверяем детали этикетки", detail: "Проверяем расположение рисунков и надписей на похожих бутылках." },
  superpoint_done: { heading: "Визуальная проверка завершена", detail: "Готовим совпадения для сравнения с распознанным текстом." },
  superpoint_failed: { heading: "Уточняем по другим признакам", detail: "Проверка деталей недоступна. Продолжаем по оставшимся признакам." },
  ocr_rescue_started: { heading: "Перепроверяем находки по тексту", detail: "Проверяем фото вин, найденных по надписям на этикетке." },
  ocr_rescue_done: { heading: "Дополнительная проверка завершена", detail: "Все доступные совпадения готовы к итоговому сравнению." },
  ocr_rescue_failed: { heading: "Продолжаем с проверенными винами", detail: "Дополнительная проверка недоступна. Используем уже найденные совпадения." },
  rank: { heading: "Выбираем лучшее совпадение", detail: "Сравниваем найденные вина между собой." },
  fusion: { heading: "Объединяем фото и текст", detail: "Сопоставляем визуальные совпадения с надписями на этикетке." },
  color: { heading: "Сравниваем цвета этикеток", detail: "Учитываем сходство цветовой палитры снимка и эталона." },
  vintage: { heading: "Сверяем год урожая", detail: "Проверяем прочитанный год по данным найденных вин." },
  done: { heading: "Вино найдено", detail: "Готовим карточку с описанием и ссылкой на сайт." },
  failed: { heading: "Поиск не завершился", detail: "Не удалось закончить распознавание. Можно попробовать другое фото." },
};

export function progressCopy(phase: RecognitionPhase, stage: JobStage | null): ProgressCopy {
  if (phase === "uploading") return {
    heading: "Отправляем снимок", detail: "Передаём фотографию на распознавание.",
  };
  if (!stage) return {
    heading: "Ваше фото в очереди", detail: "Ждём свободного сканера. Поиск начнётся автоматически.",
  };
  // A future server stage should degrade safely while client/server update.
  return STAGE_COPY[stage] ?? {
    heading: "Продолжаем распознавание", detail: "Сравниваем снимок с каталогом вин.",
  };
}
