import type { WineCandidate } from "#shared/types/recognition";

export function percent(score: number): number {
  return Math.max(0, Math.min(100, Math.round(Number(score) * 100)));
}

export function formatMb(bytes: number): string {
  return (bytes / (1024 * 1024)).toLocaleString("ru-RU", { maximumFractionDigits: 1 });
}

export function formatScore(value: number): string {
  return Number.isFinite(value) ? value.toFixed(4) : "—";
}

export function plural(count: number, forms: [string, string, string]): string {
  const mod10 = count % 10;
  const mod100 = count % 100;
  if (mod10 === 1 && mod100 !== 11) return forms[0];
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)) return forms[1];
  return forms[2];
}

export function prettifySlug(slug: string): string {
  const text = slug.replace(/[-_]+/g, " ").trim();
  return text ? text[0]!.toUpperCase() + text.slice(1) : "Без названия";
}

export interface WineView {
  name: string;
  winery: string;
  region: string;
  grape: string;
  color: string;
}

/** Бэкенд пока отдаёт только wine_id/slug; поля карточки — если придут. */
export function wineView(candidate: WineCandidate): WineView {
  return {
    name: candidate.name || prettifySlug(candidate.slug || candidate.wine_id),
    winery: candidate.winery ?? "",
    region: candidate.region ?? "",
    grape: candidate.grape_variety ?? "",
    color: candidate.color ?? "",
  };
}
