<script setup lang="ts">
import type { SearchResponse } from "#shared/types/recognition";
const props = defineProps<{ response: SearchResponse; previewUrl: string | null }>();
const emit = defineEmits<{ pick: [] }>();
const titleEl = ref<HTMLElement>();
const comparisonOpen = ref(false);
const comparisonEl = ref<HTMLElement>();
function toggleComparison() {
  if (!props.previewUrl) return;
  comparisonOpen.value = !comparisonOpen.value;
  if (comparisonOpen.value) nextTick(() => comparisonEl.value?.scrollIntoView({
    block: "nearest", behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth",
  }));
}
const wine = computed(() => props.response.candidates[0]!);
const uncertain = computed(() => props.response.decision?.accepted === false || props.response.status === "no_results");
function scorePercent(score: number) {
  return Number.isFinite(score) ? Math.round(Math.max(0, Math.min(1, score)) * 100) : null;
}
const matchPercent = computed(() => {
  const score = props.response.decision?.score ?? wine.value.score;
  return scorePercent(score);
});
const alternatives = computed(() => {
  if (!uncertain.value) return [];
  const seen = new Set([wine.value.slug]);
  return props.response.candidates.filter((candidate) => {
    if (seen.has(candidate.slug)) return false;
    seen.add(candidate.slug);
    return true;
  }).slice(0, 9).map((candidate, index) => ({
    candidate, rank: index + 2,
    name: candidate.name || prettifySlug(candidate.slug),
    url: `https://vino-svoe.ru/wines/${encodeURIComponent(candidate.slug)}`,
    percent: scorePercent(candidate.score),
    facts: [candidate.color, candidate.vintage?.toString()].filter(Boolean).join(" · "),
  }));
});
const uncertaintyMessage = computed(() => props.response.status === "ok" && props.response.message
  ? props.response.message
  : "Неуверенное совпадение: это самый близкий вариант из каталога. Сверьте название, рисунок этикетки и год.");
const name = computed(() => wine.value.name || prettifySlug(wine.value.slug));
const wineUrl = computed(() => `https://vino-svoe.ru/wines/${encodeURIComponent(wine.value.slug)}`);
const paragraphs = computed(() => wine.value.description?.trim().split(/\n\s*\n/).filter(Boolean) ?? []);
const facts = computed(() => [
  { label: "Тип", value: wine.value.category }, { label: "Цвет", value: wine.value.color },
  { label: "Регион", value: wine.value.region }, { label: "Виноград", value: wine.value.grape_variety },
  { label: "Год урожая", value: wine.value.vintage?.toString() }, ...(wine.value.attributes ?? []),
].filter((fact) => fact.value));
const vintageNote = computed(() => {
  const check = props.response.vintage_check;
  if (check?.state === "matched") return `Год на этикетке совпадает: ${check.detected_year}.`;
  if (check?.state === "unverified") return `На фото прочитан ${check.detected_year} год, но в карточке каталога год не указан.`;
  if (check?.state === "mismatch") return `На фото прочитан ${check.detected_year} год, в карточке — ${wine.value.vintage}. Проверьте винтаж.`;
  if (check?.state === "ambiguous") return "На фото несколько годов. Проверьте год урожая на этикетке.";
  return "";
});
onMounted(() => {
  window.scrollTo({ top: 0 });
  nextTick(() => titleEl.value?.focus({ preventScroll: true }));
});
</script>
<template>
  <section class="view result-view" aria-labelledby="resultTitle">
    <div class="result-head">
      <button v-if="previewUrl" class="query-chip query-toggle" type="button"
        :aria-expanded="comparisonOpen" aria-controls="photoComparison" @click="toggleComparison">
        <img :src="previewUrl" alt="">
        <span class="query-toggle__label">
          <strong>{{ comparisonOpen ? 'Закрыть сравнение' : 'Сравнить фото' }}</strong>
          <span>{{ comparisonOpen ? 'Вернуться к карточке' : 'Ваш снимок и фото из каталога' }}</span>
        </span>
        <svg class="query-toggle__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true">
          <circle cx="10" cy="10" r="6" /><path d="m15 15 5 5M7 10h6" />
          <path v-if="!comparisonOpen" d="M10 7v6" />
        </svg>
      </button>
      <button class="text-btn" type="button" @click="emit('pick')">Другое фото</button>
    </div>
    <article class="result-card" :class="{ 'result-card--comparison': comparisonOpen }">
      <div id="photoComparison" ref="comparisonEl" class="result-card__images" :class="{ 'result-card__images--expanded': comparisonOpen }">
        <figure class="comparison-query" :aria-hidden="!comparisonOpen">
          <figcaption>Ваш снимок</figcaption>
          <div class="query-photo__frame"><img v-if="previewUrl" :src="previewUrl" alt="Ваш снимок этикетки для сравнения"></div>
        </figure>
        <figure class="comparison-reference">
          <figcaption>Фото из каталога</figcaption>
          <button class="result-card__photo" type="button" :disabled="!previewUrl"
            :aria-label="comparisonOpen ? 'Скрыть сравнение снимков' : 'Сравнить ваш снимок с фото из каталога'"
            :aria-expanded="comparisonOpen" aria-controls="photoComparison" @click="toggleComparison">
            <WineImage :candidate="wine" :name="name" />
          </button>
        </figure>
      </div>
      <div class="result-card__body">
        <Transition name="comparison-note"><p v-if="comparisonOpen" class="comparison-note">Сравните название, рисунок этикетки и год на двух снимках.</p></Transition>
        <p class="eyebrow">Похоже, это</p>
        <h1 id="resultTitle" ref="titleEl" tabindex="-1">{{ name }}</h1>
        <p v-if="wine.winery" class="result-winery">{{ wine.winery }}</p>
        <div v-if="matchPercent !== null" class="match-quality" :class="{ 'match-quality--uncertain': uncertain }">
          <p class="match-quality__score">Оценка совпадения: <strong>{{ matchPercent }}%</strong></p>
          <p class="match-quality__hint">Оценка модели, не вероятность правильного ответа.</p>
        </div>
        <p v-if="uncertain" class="match-warning" role="status">{{ uncertaintyMessage }}</p>
        <p class="result-caution">Сверьте название и год с вашей этикеткой.</p>
        <a class="primary-link" :href="wineUrl" target="_blank" rel="noopener noreferrer">
          <span>Открыть на Вино своё</span><span aria-hidden="true">↗</span>
        </a>
        <p v-if="response.warnings?.length" class="result-notice" role="status">
          Не все этапы распознавания отработали. Проверьте найденное вино на сайте.
        </p>
        <p v-if="vintageNote" class="vintage-note" :class="{ 'vintage-note--warning': response.vintage_check?.state === 'mismatch' }">{{ vintageNote }}</p>
        <dl v-if="facts.length" class="wine-facts">
          <div v-for="fact in facts" :key="fact.label"><dt>{{ fact.label }}</dt><dd>{{ fact.value }}</dd></div>
        </dl>
      </div>
    </article>
    <section v-if="alternatives.length" class="candidate-options" aria-labelledby="candidateOptionsTitle">
      <h2 id="candidateOptionsTitle">Другие похожие варианты</h2>
      <p class="candidate-options__hint">Совпадение неоднозначное. Вместе с основным вариантом выше показываем до 10 кандидатов — сравните их этикетки и год.</p>
      <p class="candidate-options__hint">Проценты — оценки совпадения, не вероятности. Отзыв ниже относится к основному варианту.</p>
      <ol class="candidate-options__list" start="2">
        <li v-for="option in alternatives" :key="option.candidate.slug">
          <a class="candidate-option" :href="option.url" target="_blank" rel="noopener noreferrer"
            :aria-label="`Вариант ${option.rank}: ${option.name}. Открыть на Вино своё в новой вкладке`">
            <div class="candidate-option__photo"><WineImage :candidate="option.candidate" :name="option.name" loading="lazy" /></div>
            <div class="candidate-option__body">
              <span class="candidate-option__rank">Вариант {{ option.rank }}</span>
              <h3>{{ option.name }}</h3>
              <p v-if="option.candidate.winery" class="candidate-option__winery">{{ option.candidate.winery }}</p>
              <p v-if="option.facts" class="candidate-option__facts">{{ option.facts }}</p>
              <div class="candidate-option__footer">
                <span v-if="option.percent !== null">Совпадение: {{ option.percent }}%</span>
                <span class="candidate-option__link">На сайт <span aria-hidden="true">↗</span></span>
              </div>
            </div>
          </a>
        </li>
      </ol>
    </section>
    <RecognitionFeedback :key="response.request_id" :job-id="response.request_id" />
    <section v-if="paragraphs.length" class="wine-description" aria-labelledby="descriptionTitle">
      <h2 id="descriptionTitle">О вине</h2>
      <p v-for="(paragraph, index) in paragraphs" :key="index">{{ paragraph }}</p>
      <span class="catalog-credit">Описание из каталога «Вино своё»</span>
    </section>
  </section>
</template>
