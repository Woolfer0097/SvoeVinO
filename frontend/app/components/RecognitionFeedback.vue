<script setup lang="ts">
const props = defineProps<{ jobId: string; unrecognized?: boolean }>();
type IndexState = "queued" | "indexing" | "ready" | "failed" | "unlabeled";
const indexState = ref<IndexState>("unlabeled");
let monitor: AbortController | null = null;
const savedMessage = computed(() => indexState.value === "ready"
  ? "Спасибо! Фото добавлено в поиск и поможет распознавать это вино."
  : indexState.value === "queued" || indexState.value === "indexing"
    ? "Отзыв сохранён. Добавляем фото в поиск…"
    : indexState.value === "failed" ? "Отзыв сохранён, но фото пока не добавлено в поиск. Повторите сохранение позже."
    : "Отзыв сохранён. Укажите правильное вино, чтобы добавить фото в поиск.");
const verdict = ref<boolean | null>(props.unrecognized ? false : null);
const detailsOpen = ref(!!props.unrecognized);
const information = reactive<{ name: string; year: string | number; locality: string; winery: string; notes: string }>(
  { name: "", year: "", locality: "", winery: "", notes: "" });
const hasDetails = computed(() => Object.values(information).some(value => String(value).trim()));
const notice = ref("");
const busy = ref(false);
const saved = ref(false);
const error = ref("");
function choose(value: boolean) {
  verdict.value = value;
  saved.value = false;
  error.value = "";
  notice.value = "";
  if (value) void save(true);
  else {
    monitor?.abort();
    detailsOpen.value = true;
    nextTick(() => document.getElementById("feedbackWineName")?.focus());
    // Retract a previous confirmation if the user changes their mind. A first
    // negative vote without details does not persist a new photo.
    void save(true);
  }
}
async function save(confirmOnly = false) {
  if (verdict.value === null || busy.value) return;
  busy.value = true;
  error.value = "";
  try {
    monitor?.abort();
    const response = await $fetch<{ saved: boolean; message?: string; embedding_state: IndexState }>("/api/feedback", { method: "POST", retry: 0, timeout: 100_000, body: {
      job_id: props.jobId, is_correct: verdict.value,
      wine_information: confirmOnly ? null : {
        ...information, year: String(information.year).trim() ? Number(information.year) : null,
      },
    } });
    saved.value = response.saved;
    notice.value = response.saved ? "" : response.message || "Уточните сведения о вине.";
    if (!response.saved) return;
    indexState.value = response.embedding_state;
    if (indexState.value === "queued" || indexState.value === "indexing") void watchIndex();
  } catch (failure) {
    const detail = failure as { data?: { message?: string } };
    error.value = detail.data?.message || "Не удалось сохранить отзыв. Попробуйте ещё раз.";
  } finally { busy.value = false; }
}
async function watchIndex() {
  const current = new AbortController();
  monitor = current;
  const started = Date.now();
  try {
    while (!current.signal.aborted && Date.now() - started < 120_000) {
      await new Promise(resolve => setTimeout(resolve, 1500));
      if (current.signal.aborted) return;
      const response = await $fetch<{ embedding_state: IndexState }>("/api/feedback/status", {
        query: { job_id: props.jobId }, retry: 0, signal: current.signal,
      });
      indexState.value = response.embedding_state;
      if (indexState.value !== "queued" && indexState.value !== "indexing") return;
    }
  } catch { /* The vote is already durable; a lost status request must not erase it. */ }
}
onBeforeUnmount(() => monitor?.abort());
</script>
<template>
  <section class="recognition-feedback" aria-labelledby="feedbackTitle">
    <h2 id="feedbackTitle">{{ unrecognized ? 'Знаете это вино?' : 'Это ваше вино?' }}</h2>
    <p class="feedback-hint">Расскажите, что знаете о вине: название, год урожая, где оно произведено и какая винодельня его создала. Можно заполнить только знакомые вам сведения.</p>
    <div class="feedback-actions">
      <button v-if="!unrecognized" class="feedback-choice" :aria-pressed="verdict === true" :disabled="busy" @click="choose(true)">Угадал</button>
      <button class="feedback-choice" :aria-pressed="verdict === false" :disabled="busy" @click="choose(false)">{{ unrecognized ? 'Рассказать о вине' : 'Не угадал' }}</button>
    </div>
    <button v-if="!detailsOpen" class="text-btn feedback-more" type="button" :disabled="busy"
      @click="detailsOpen = true; verdict ??= true">Добавить сведения о вине</button>
    <form v-if="detailsOpen" class="feedback-form" @submit.prevent="save()" @input="error = ''; notice = ''; saved = false">
      <label for="feedbackWineName">Название вина</label>
      <input id="feedbackWineName" v-model="information.name" type="text" maxlength="200" autocomplete="off"
        placeholder="Как написано на этикетке" :disabled="busy">
      <div class="feedback-fields">
        <div><label for="feedbackYear">Год урожая</label><input id="feedbackYear" v-model="information.year"
          type="number" inputmode="numeric" min="1900" max="2099" placeholder="Например, 2021" :disabled="busy"></div>
        <div><label for="feedbackLocality">Город или регион</label><input id="feedbackLocality" v-model="information.locality"
          type="text" maxlength="200" autocomplete="off" placeholder="Где произведено" :disabled="busy"></div>
      </div>
      <label for="feedbackWinery">Винодельня</label>
      <input id="feedbackWinery" v-model="information.winery" type="text" maxlength="200" autocomplete="off" :disabled="busy">
      <label for="feedbackNotes">Что ещё знаете о вине</label>
      <textarea id="feedbackNotes" v-model="information.notes" maxlength="1500" rows="3"
        placeholder="Сорт винограда, цвет, текст этикетки или любые детали, которыми хотите поделиться" :disabled="busy" />
      <button class="feedback-choice" type="submit" :disabled="busy || !hasDetails">{{ busy ? 'Ищем вино по вашим сведениям…' : 'Отправить сведения' }}</button>
    </form>
    <p v-if="notice" class="feedback-message" role="status">{{ notice }}</p>
    <p v-if="saved" class="feedback-message" role="status">{{ savedMessage }}</p>
    <button v-if="saved && indexState === 'failed'" class="text-btn" :disabled="busy" @click="save(verdict === true && !hasDetails)">Повторить добавление в поиск</button>
    <p v-if="error" class="feedback-message feedback-error" role="alert">{{ error }}</p>
    <button v-if="error && verdict === true" class="text-btn" :disabled="busy" @click="save(!hasDetails)">Повторить</button>
  </section>
</template>
<style scoped>
.recognition-feedback { margin-top: 24px; padding: 24px; border: 1px solid #ded4c8; border-radius: 22px; background: #fffdf8; }
h2 { margin: 0; font-size: 20px; }
.feedback-hint, .feedback-message { font-size: 14px; line-height: 1.6; }
.feedback-hint { color: #71665c; margin: 8px 0 18px; }
.feedback-actions { display: flex; gap: 10px; flex-wrap: wrap; }
.feedback-choice { min-height: 44px; border: 1px solid #bda999; padding: 10px 20px; border-radius: 12px; background: transparent; color: #652c3c; cursor: pointer; font: inherit; }
.feedback-choice[aria-pressed="true"] { background: #652c3c; color: white; border-color: #652c3c; }
.feedback-choice:disabled { opacity: .55; cursor: wait; }
.feedback-choice:focus-visible, input:focus-visible, textarea:focus-visible { outline: 3px solid #9b5365; outline-offset: 3px; }
.feedback-form { display: grid; gap: 12px; margin-top: 20px; }
.feedback-form label { font-size: 14px; line-height: 1.5; }
.feedback-form label span { color: #71665c; }
.feedback-form input, .feedback-form textarea { min-width: 0; width: 100%; box-sizing: border-box; border: 1px solid #bda999; background: white; border-radius: 10px; padding: 12px; font: inherit; font-size: 16px; }
.feedback-form textarea { resize: vertical; }
.feedback-fields { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 2fr); gap: 12px; }
.feedback-fields > div { min-width: 0; display: grid; gap: 8px; }
.feedback-more { display: inline-block; margin-top: 18px; }
@media (max-width: 370px) { .feedback-fields { grid-template-columns: minmax(0, 1fr); } }
.feedback-form button { justify-self: start; }
.feedback-error { color: #9c203c; }
</style>
