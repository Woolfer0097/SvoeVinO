/*
 * Демо-каталог мок-сервера. Названия вин и виноделен вымышлены; регионы и
 * сорта винограда — настоящие.
 */
import type { SearchResponse, WineCandidate } from "#shared/types/recognition";
import { hashString, mulberry32 } from "#shared/utils/hash";

export const MOCK_MODEL_NAME = "facebook/dinov2-with-registers-giant";

type MockWine = Required<Pick<WineCandidate, "slug" | "name" | "winery" | "region" | "grape_variety" | "color">>;

const MOCK_WINES: MockWine[] = [
  { slug: "solnechnyj-ustup-zakatnyj-veter", name: "Закатный ветер", winery: "Винодельня «Солнечный уступ»", region: "Краснодарский край", grape_variety: "Красностоп золотовский", color: "Красное сухое" },
  { slug: "tri-kurgana-melovaya-tropa", name: "Меловая тропа", winery: "Усадьба «Три кургана»", region: "Ростовская область", grape_variety: "Цимлянский чёрный", color: "Красное сухое" },
  { slug: "tikhaya-zavod-yantarnyj-bereg", name: "Янтарный берег", winery: "Хутор «Тихая заводь»", region: "Республика Крым", grape_variety: "Кокур белый", color: "Белое сухое" },
  { slug: "vetryanaya-gora-polden", name: "Полдень", winery: "Дом «Ветряная гора»", region: "Севастополь", grape_variety: "Шардоне", color: "Белое сухое" },
  { slug: "staryj-brod-rozovyj-sad", name: "Розовый сад", winery: "Погреба «Старый брод»", region: "Краснодарский край", grape_variety: "Пино нуар", color: "Розовое сухое" },
  { slug: "melovoj-sklon-serebryanaya-loza", name: "Серебряная лоза", winery: "Винодельня «Меловой склон»", region: "Ростовская область", grape_variety: "Сибирьковый", color: "Белое сухое" },
  { slug: "solnechnyj-ustup-nochnoj-khutor", name: "Ночной хутор", winery: "Винодельня «Солнечный уступ»", region: "Краснодарский край", grape_variety: "Каберне совиньон", color: "Красное сухое" },
  { slug: "vetryanaya-gora-pervyj-sneg", name: "Первый снег", winery: "Дом «Ветряная гора»", region: "Севастополь", grape_variety: "Шардоне, пино нуар", color: "Игристое брют" },
  { slug: "kamennyj-bereg-dikij-mindal", name: "Дикий миндаль", winery: "Хозяйство «Каменный берег»", region: "Республика Крым", grape_variety: "Мускат белый", color: "Белое полусладкое" },
  { slug: "tri-kurgana-stepnaya-zvezda", name: "Степная звезда", winery: "Усадьба «Три кургана»", region: "Ростовская область", grape_variety: "Красностоп золотовский", color: "Красное сухое" },
  { slug: "gornyj-rodnik-ognennyj-kamen", name: "Огненный камень", winery: "Винодельня «Горный родник»", region: "Республика Дагестан", grape_variety: "Саперави", color: "Красное сухое" },
  { slug: "tikhaya-zavod-oranzhevyj-chas", name: "Оранжевый час", winery: "Хутор «Тихая заводь»", region: "Республика Крым", grape_variety: "Ркацители", color: "Оранжевое сухое" },
  { slug: "staryj-brod-vesennij-sad", name: "Весенний сад", winery: "Погреба «Старый брод»", region: "Краснодарский край", grape_variety: "Рислинг", color: "Белое сухое" },
  { slug: "sosnovyj-mys-utrennyaya-rosa", name: "Утренняя роса", winery: "Винодельня «Сосновый мыс»", region: "Краснодарский край", grape_variety: "Совиньон блан", color: "Белое сухое" },
  { slug: "gornyj-rodnik-vysokoe-nebo", name: "Высокое небо", winery: "Винодельня «Горный родник»", region: "Республика Дагестан", grape_variety: "Ркацители", color: "Белое сухое" },
  { slug: "kamennyj-bereg-morskaya-glad", name: "Морская гладь", winery: "Хозяйство «Каменный берег»", region: "Республика Крым", grape_variety: "Мерло", color: "Розовое сухое" },
  { slug: "sosnovyj-mys-tikhij-shtorm", name: "Тихий шторм", winery: "Винодельня «Сосновый мыс»", region: "Краснодарский край", grape_variety: "Мерло", color: "Красное сухое" },
  { slug: "melovoj-sklon-donskoj-vecher", name: "Донской вечер", winery: "Винодельня «Меловой склон»", region: "Ростовская область", grape_variety: "Цимлянский чёрный", color: "Красное сухое" },
  { slug: "vetryanaya-gora-belyj-parus", name: "Белый парус", winery: "Дом «Ветряная гора»", region: "Севастополь", grape_variety: "Совиньон блан", color: "Белое сухое" },
  { slug: "verkhnij-lug-lugovaya-pesnya", name: "Луговая песня", winery: "Усадьба «Верхний луг»", region: "Ставропольский край", grape_variety: "Мускат белый", color: "Игристое полусладкое" },
  { slug: "verkhnij-lug-temnyj-les", name: "Тёмный лес", winery: "Усадьба «Верхний луг»", region: "Ставропольский край", grape_variety: "Сира", color: "Красное сухое" },
  { slug: "staryj-brod-rubinovaya-nit", name: "Рубиновая нить", winery: "Погреба «Старый брод»", region: "Краснодарский край", grape_variety: "Красностоп золотовский, каберне совиньон", color: "Красное сухое" },
  { slug: "tikhaya-zavod-lunnyj-sad", name: "Лунный сад", winery: "Хутор «Тихая заводь»", region: "Республика Крым", grape_variety: "Пино нуар", color: "Игристое брют" },
  { slug: "solnechnyj-ustup-medovyj-sklon", name: "Медовый склон", winery: "Винодельня «Солнечный уступ»", region: "Краснодарский край", grape_variety: "Шардоне", color: "Белое сухое" },
];

const round4 = (value: number) => Math.round(value * 10000) / 10000;

/** Одинаковый seed (имя и размер фото) всегда даёт одинаковый ответ. */
export function buildMockResult(requestId: string, seed: string, topK: number): SearchResponse {
  const random = mulberry32(hashString(seed));
  const pool = MOCK_WINES.slice();
  for (let i = pool.length - 1; i > 0; i -= 1) {
    const j = Math.floor(random() * (i + 1));
    [pool[i], pool[j]] = [pool[j]!, pool[i]!];
  }

  let score = 0.86 + random() * 0.08;
  const candidates = pool.slice(0, Math.min(topK, pool.length)).map((wine, index): WineCandidate => {
    if (index === 1) score -= 0.05 + random() * 0.05;
    else if (index > 1) score -= 0.006 + random() * 0.02;
    const value = Math.max(0.35, score);
    return {
      ...wine,
      wine_id: wine.slug,
      score: round4(value),
      distance: round4(1 - value),
      best_image_uri: `/data/reference/${wine.slug}/front.webp`,
    };
  });

  return {
    request_id: requestId,
    status: "ok",
    model_name: MOCK_MODEL_NAME,
    query_embedding_dimension: 1536,
    candidates,
  };
}

export function buildEmptyResult(requestId: string): SearchResponse {
  return {
    request_id: requestId,
    status: "no_results",
    model_name: MOCK_MODEL_NAME,
    query_embedding_dimension: 1536,
    candidates: [],
    message: `No reference photos are indexed for model ${MOCK_MODEL_NAME}; run the index command first`,
  };
}
