<script setup lang="ts">
import type { JobStage } from "#shared/types/recognition";
import type { RecognitionPhase } from "~/composables/useRecognition";

const props = defineProps<{
  previewUrl: string | null;
  phase: RecognitionPhase;
  stage: JobStage | null;
  progress: number;
  jobId: string | null;
  pollIntervalMs: number;
}>();
const emit = defineEmits<{ cancel: [] }>();

const STEPS = ["Загружаем фото", "Готовим изображение", "Сравниваем с каталогом", "Собираем похожие вина"];
const STAGE_STEP: Record<JobStage, number> = { prepare: 1, search: 2, rank: 3 };

// Этапы приходят из статуса задачи; очередь показываем как начало подготовки.
const activeStep = computed(() => {
  if (props.phase === "uploading") return 0;
  return props.stage ? STAGE_STEP[props.stage] : 1;
});

const heading = computed(() => {
  if (props.phase === "uploading") return "Загружаем фото…";
  if (!props.stage) return "Фото в очереди…";
  return `${STEPS[activeStep.value]}…`;
});

const percentDone = computed(() => Math.round(Math.max(props.phase === "uploading" ? 0.04 : 0.08, props.progress) * 100));
const pollSeconds = computed(() => (props.pollIntervalMs / 1000).toLocaleString("ru-RU"));
</script>

<template>
  <section class="view" aria-labelledby="loadingStage">
    <div class="scan">
      <div class="scan-frame">
        <img v-if="previewUrl" :src="previewUrl" alt="Ваше фото">
        <span class="scan-line" aria-hidden="true" />
        <span class="corner corner--tl" aria-hidden="true" />
        <span class="corner corner--tr" aria-hidden="true" />
        <span class="corner corner--bl" aria-hidden="true" />
        <span class="corner corner--br" aria-hidden="true" />
      </div>

      <div class="scan-status">
        <p class="eyebrow">Идёт поиск</p>
        <h2 id="loadingStage" aria-live="polite">{{ heading }}</h2>

        <div
          class="scan-progress"
          role="progressbar"
          aria-label="Прогресс поиска"
          aria-valuemin="0"
          aria-valuemax="100"
          :aria-valuenow="percentDone"
        >
          <span :style="{ width: `${percentDone}%` }" />
        </div>

        <ol class="stage-list">
          <li
            v-for="(label, index) in STEPS"
            :key="label"
            :class="{ 'is-done': index < activeStep, 'is-active': index === activeStep }"
          >
            {{ label }}
          </li>
        </ol>

        <p v-if="jobId" class="scan-meta">
          Задача <code>{{ jobId.slice(0, 8) }}</code> · статус обновляется примерно раз в {{ pollSeconds }} с
        </p>
        <button class="text-btn" type="button" @click="emit('cancel')">Отменить</button>
      </div>
    </div>
  </section>
</template>
