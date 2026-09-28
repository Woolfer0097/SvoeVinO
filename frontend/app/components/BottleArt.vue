<script setup lang="ts">
import type { BottleWine } from "~/utils/bottle";

const props = defineProps<{ wine: BottleWine; label?: string }>();

// Уникальный id градиента: на странице одновременно много бутылок.
const gradientId = `bottle-${useId().replace(/[^\w-]/g, "")}-glass`;
const palette = computed(() => bottlePalette(props.wine));
const ornament = computed(() => bottleOrnament(props.wine));
const letter = computed(() => {
  const title = (props.wine.name || props.wine.slug || props.wine.wine_id || "?").trim();
  return (title[0] ?? "?").toUpperCase();
});
</script>

<template>
  <svg viewBox="0 0 200 264" role="img" :aria-label="label || 'Иллюстрация бутылки'">
    <defs>
      <linearGradient :id="gradientId" x1="0" x2="1" y1="0" y2="0">
        <stop offset="0" :stop-color="palette.glass[1]" />
        <stop offset=".3" :stop-color="palette.glass[0]" />
        <stop offset=".78" :stop-color="palette.glass[0]" />
        <stop offset="1" :stop-color="palette.glass[1]" />
      </linearGradient>
    </defs>
    <ellipse cx="100" cy="252" rx="48" ry="6" fill="#000" opacity=".12" />
    <path :d="BOTTLE_PATH" :fill="`url(#${gradientId})`" />
    <rect x="72" y="108" width="6" height="124" rx="3" fill="#fff" opacity=".16" />
    <rect x="91" y="50" width="3" height="24" rx="1.5" fill="#fff" opacity=".2" />
    <rect x="86" y="10" width="28" height="40" rx="4" :fill="palette.foil" />
    <rect x="86" y="44" width="28" height="3" fill="#000" opacity=".15" />
    <rect x="72" y="138" width="56" height="74" rx="5" :fill="palette.label" :stroke="palette.accent" stroke-width="1" />
    <rect x="76" y="142" width="48" height="66" rx="3" fill="none" :stroke="palette.accent" stroke-width=".6" opacity=".7" />

    <circle v-if="ornament === 0" cx="100" cy="158" r="6" fill="none" :stroke="palette.accent" stroke-width="1.2" />
    <path v-else-if="ornament === 1" d="M100 150L106 158L100 166L94 158Z" fill="none" :stroke="palette.accent" stroke-width="1.2" />
    <g v-else-if="ornament === 2" :fill="palette.accent">
      <circle cx="97" cy="156" r="2.6" />
      <circle cx="103" cy="156" r="2.6" />
      <circle cx="100" cy="161" r="2.6" />
      <path d="M100 153V149" :stroke="palette.accent" stroke-width="1" />
    </g>
    <g v-else :stroke="palette.accent" stroke-width="1" stroke-linecap="round">
      <circle cx="100" cy="158" r="3" :fill="palette.accent" stroke="none" />
      <path d="M100 149V151M100 165V167M91 158H93M107 158H109M94 152L95.5 153.5M104.5 162.5L106 164M94 164L95.5 162.5M104.5 153.5L106 152" />
    </g>

    <text x="100" y="187" text-anchor="middle" font-family="Cormorant Garamond, Georgia, serif" font-size="23" font-weight="700" :fill="palette.ink">{{ letter }}</text>
    <line x1="84" y1="196" x2="116" y2="196" :stroke="palette.accent" stroke-width=".8" />
    <line x1="90" y1="201" x2="110" y2="201" :stroke="palette.accent" stroke-width=".8" opacity=".6" />
  </svg>
</template>
