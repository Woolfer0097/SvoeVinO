<script setup lang="ts">
import type { JobStage } from "#shared/types/recognition";
import type { RecognitionPhase } from "~/composables/useRecognition";
import { progressCopy } from "~/utils/progress";
const props = defineProps<{
  previewUrl: string | null; phase: RecognitionPhase; stage: JobStage | null;
  progress: number; jobId: string | null; pollIntervalMs: number;
}>();
const emit = defineEmits<{ cancel: [] }>();
const copy = computed(() => progressCopy(props.phase, props.stage));
const copyKey = computed(() => `${props.phase}:${props.stage ?? 'queued'}`);
const percentDone = computed(() => Math.round(Math.max(.05, props.progress) * 100));
</script>
<template>
  <section class="view" aria-labelledby="loadingStage">
    <div class="recognition-progress">
      <div class="scan-preview"><img v-if="previewUrl" :src="previewUrl" alt="Ваш снимок этикетки"><span class="scan-line" aria-hidden="true" /></div>
      <p class="eyebrow">Немного терпения</p>
      <div class="stage-announcement" role="status" aria-live="polite" aria-atomic="true">
        <Transition name="stage-copy" mode="out-in">
          <div :key="copyKey" class="stage-copy">
            <h1 id="loadingStage">{{ copy.heading }}…</h1>
            <p>{{ copy.detail }}</p>
          </div>
        </Transition>
      </div>
      <div class="scan-progress" role="progressbar" aria-label="Прогресс распознавания" aria-valuemin="0" aria-valuemax="100" :aria-valuenow="percentDone">
        <span :style="{ width: `${percentDone}%` }" />
      </div>
      <button class="text-btn" type="button" @click="emit('cancel')">Отменить</button>
    </div>
  </section>
</template>
