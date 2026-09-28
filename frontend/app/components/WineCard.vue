<script setup lang="ts">
import type { WineCandidate } from "#shared/types/recognition";

const props = defineProps<{ candidate: WineCandidate; rank: number }>();
const emit = defineEmits<{ select: [] }>();

const view = computed(() => wineView(props.candidate));
const pct = computed(() => percent(props.candidate.score));
const meta = computed(() => view.value.winery || view.value.region || props.candidate.slug);
const tint = computed(() => bottlePalette(props.candidate).tint);
</script>

<template>
  <button
    class="wine-card"
    type="button"
    :aria-label="`${view.name}, сходство ${pct}%. Показать подробнее`"
    @click="emit('select')"
  >
    <span class="wine-card__media" :style="{ '--tint': tint }">
      <WineImage :candidate="candidate" :name="view.name" />
    </span>
    <span class="wine-card__rank">№{{ rank }}</span>
    <span class="wine-card__name">{{ view.name }}</span>
    <span class="wine-card__meta">{{ meta }}</span>
    <span class="wine-card__meter">
      <span class="wine-card__bar"><span :style="{ '--w': `${pct}%` }" /></span>
      <span class="wine-card__pct">{{ pct }}%</span>
    </span>
  </button>
</template>
