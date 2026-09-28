<script setup lang="ts">
import type { SearchResponse } from "#shared/types/recognition";

const props = defineProps<{ response: SearchResponse; previewUrl: string | null }>();
const emit = defineEmits<{ pick: [] }>();

const featuredIndex = ref(0);
const featuredEl = ref<HTMLElement>();
const titleEl = ref<HTMLElement>();
const swapKey = ref(0);

const featured = computed(() => props.response.candidates[featuredIndex.value]!);
const view = computed(() => wineView(featured.value));
const pct = computed(() => percent(featured.value.score));
const tint = computed(() => bottlePalette(featured.value).tint);
const chips = computed(() => [view.value.color, view.value.grape, view.value.region].filter(Boolean));
const others = computed(() =>
  props.response.candidates
    .map((candidate, index) => ({ candidate, index }))
    .filter(({ index }) => index !== featuredIndex.value),
);

function focusTitle() {
  nextTick(() => titleEl.value?.focus({ preventScroll: true }));
}

function select(index: number) {
  featuredIndex.value = index;
  swapKey.value += 1;
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  featuredEl.value?.scrollIntoView({ behavior: reduced ? "auto" : "smooth", block: "start" });
  focusTitle();
}

watch(
  () => props.response,
  () => {
    featuredIndex.value = 0;
  },
);

onMounted(() => {
  window.scrollTo({ top: 0 });
  focusTitle();
});
</script>

<template>
  <section class="view" aria-labelledby="resultTitle">
    <div class="result-head">
      <div class="query-chip">
        <img v-if="previewUrl" :src="previewUrl" alt="">
        <span>Ваше фото</span>
      </div>
      <button class="ghost-btn" type="button" @click="emit('pick')">Проверить другое фото</button>
    </div>

    <article :key="swapKey" ref="featuredEl" class="featured" :class="{ 'is-swapping': swapKey > 0 }">
      <div class="featured__media" :style="{ '--tint': tint }">
        <WineImage :candidate="featured" :name="view.name" />
        <span v-if="featuredIndex === 0" class="featured__badge">Лучшее совпадение</span>
      </div>

      <div class="featured__body">
        <p class="eyebrow">
          {{ featuredIndex === 0 ? "Больше всего похоже на" : `Вы выбрали · №${featuredIndex + 1} по сходству` }}
        </p>
        <h2 id="resultTitle" ref="titleEl" tabindex="-1">{{ view.name }}</h2>
        <p v-if="view.winery" class="featured__winery">{{ view.winery }}</p>
        <ul v-if="chips.length" class="chips">
          <li v-for="chip in chips" :key="chip">{{ chip }}</li>
        </ul>

        <div class="meter">
          <div class="meter__value"><strong>{{ pct }}</strong><span>% сходства</span></div>
          <div class="meter__bar" role="img" :aria-label="`Сходство ${pct}%`">
            <span :style="{ '--w': `${pct}%` }" />
          </div>
          <p class="meter__note">
            Насколько ваше фото похоже на эталонное фото этого вина. Это оценка сходства, а не гарантия совпадения.
          </p>
        </div>

        <details class="tech">
          <summary>Технические детали</summary>
          <dl>
            <dt>wine_id</dt><dd>{{ featured.wine_id }}</dd>
            <dt>score</dt><dd>{{ formatScore(featured.score) }}</dd>
            <dt>distance</dt><dd>{{ formatScore(featured.distance) }}</dd>
            <dt>Эталон</dt><dd>{{ featured.best_image_uri }}</dd>
            <dt>Модель</dt><dd>{{ response.model_name }} · {{ response.query_embedding_dimension }}</dd>
            <dt>request_id</dt><dd>{{ response.request_id }}</dd>
          </dl>
        </details>
      </div>
    </article>

    <div v-if="others.length">
      <div class="similar-head">
        <h3>Не то вино? Выберите из похожих</h3>
        <span class="similar-count">{{ others.length }} {{ plural(others.length, ["вино", "вина", "вин"]) }}</span>
      </div>
      <ul class="similar-grid">
        <li v-for="(item, position) in others" :key="item.candidate.wine_id" :style="{ '--i': position }">
          <WineCard :candidate="item.candidate" :rank="item.index + 1" @select="select(item.index)" />
        </li>
      </ul>
    </div>
  </section>
</template>
