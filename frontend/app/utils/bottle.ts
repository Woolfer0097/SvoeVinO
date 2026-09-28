/*
 * Иллюстрация бутылки вместо фото эталона: цвет стекла зависит от стиля вина,
 * узор этикетки — от хэша slug.
 */
import { hashString } from "#shared/utils/hash";

// Силуэт бутылки в координатах viewBox 0 0 200 264 (центр по x = 100).
export const BOTTLE_PATH =
  "M88 14H112V60C112 76 136 86 136 110V236Q136 248 124 248H76Q64 248 64 236V110C64 86 88 76 88 60Z";

export interface BottlePalette {
  glass: [string, string];
  foil: string;
  label: string;
  accent: string;
  ink: string;
  tint: string;
}

const PALETTES = {
  red: { glass: ["#2B2023", "#4E363C"], foil: "#6E1530", label: "#F6EFE3", accent: "#B8893B", ink: "#6E1530", tint: "#F3E6E9" },
  white: { glass: ["#B3B384", "#D9D7AF"], foil: "#C9A45C", label: "#FBF7EE", accent: "#B8893B", ink: "#5B4A1E", tint: "#F5F0DF" },
  rose: { glass: ["#DE9FA6", "#F3D1D3"], foil: "#B8566A", label: "#FFF9F6", accent: "#C58A94", ink: "#8C3448", tint: "#FBEDEF" },
  sparkling: { glass: ["#1F2A22", "#394C3F"], foil: "#C9A24E", label: "#F7F1E4", accent: "#B8893B", ink: "#2F3B2A", tint: "#EEF0E4" },
  orange: { glass: ["#B0702C", "#D9A15A"], foil: "#7A4A17", label: "#FBF4E8", accent: "#B8893B", ink: "#7A4A17", tint: "#F8ECDB" },
} satisfies Record<string, BottlePalette>;

type BottleStyle = keyof typeof PALETTES;
const STYLES = Object.keys(PALETTES) as BottleStyle[];

export interface BottleWine {
  slug?: string;
  wine_id?: string;
  name?: string;
  color?: string;
}

function styleOf(wine: BottleWine): BottleStyle {
  const color = (wine.color ?? "").toLowerCase();
  if (color.includes("игрист")) return "sparkling";
  if (color.includes("оранж")) return "orange";
  if (color.includes("роз")) return "rose";
  if (color.includes("бел")) return "white";
  if (color.includes("красн")) return "red";
  return STYLES[hashString(wine.slug ?? wine.wine_id ?? "") % STYLES.length]!;
}

export function bottlePalette(wine: BottleWine): BottlePalette {
  return PALETTES[styleOf(wine)];
}

/** Номер узора на этикетке: 0–3. */
export function bottleOrnament(wine: BottleWine): number {
  return hashString(wine.slug ?? wine.wine_id ?? wine.name ?? "") % 4;
}
