<script setup lang="ts">
/*
 * Единственная страница: загрузка фото → polling статуса → похожие вина.
 * ?demo=1 — поиск на сгенерированном фото; ?scenario=slow|error|empty|flaky —
 * демо-сценарии встроенного мок-сервера.
 */
import type { PhotoSource } from "~/utils/photo";

const route = useRoute();
const config = useRuntimeConfig().public;
const recognition = useRecognition();
const { phase, stage, progress, jobId, result, problem, isBusy } = recognition;

const cameraInput = ref<HTMLInputElement>();
const galleryInput = ref<HTMLInputElement>();
const hero = ref<{ focus: () => void }>();
const uploadError = ref<string | null>(null);
const previewUrl = ref<string | null>(null);

const maxMb = formatMb(config.maxBytes);
const scenario = computed(() => (typeof route.query.scenario === "string" ? route.query.scenario : undefined));

function setPreview(file: File) {
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value);
  previewUrl.value = URL.createObjectURL(file);
}

function focusUpload() {
  nextTick(() => hero.value?.focus());
}

function openPicker(source: PhotoSource = "camera") {
  const input = source === "camera" ? cameraInput.value : galleryInput.value;
  if (!input) return;
  input.value = "";
  input.click();
}

function handleFile(file: File) {
  if (isBusy.value) return;
  const invalid = validatePhoto(file, config.maxBytes);
  if (invalid) {
    recognition.reset();
    uploadError.value = invalid;
    focusUpload();
    return;
  }
  uploadError.value = null;
  setPreview(file);
  recognition.start(file, { scenario: scenario.value });
}

function onFileChange(event: Event) {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  input.value = "";
  if (file) handleFile(file);
}

function cancel() {
  recognition.cancel();
  focusUpload();
}

function retry() {
  recognition.retry({ scenario: scenario.value });
}

const { isDragging } = useFileDrop(handleFile, () => !isBusy.value);

function onKeydown(event: KeyboardEvent) {
  if (event.key === "Escape" && isBusy.value) cancel();
}

onMounted(() => {
  document.addEventListener("keydown", onKeydown);
  if (route.query.demo !== undefined) makeDemoPhoto().then(handleFile);
});

onBeforeUnmount(() => {
  document.removeEventListener("keydown", onKeydown);
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value);
});
</script>

<template>
  <div class="page" :class="{ 'page--with-bar': phase === 'done' }">
    <AppHeader :demo="config.demoData" />

    <main>
      <Transition name="view" mode="out-in">
        <ScanProgress
          v-if="isBusy"
          key="progress"
          :preview-url="previewUrl"
          :phase="phase"
          :stage="stage"
          :progress="progress"
          :job-id="jobId"
          :poll-interval-ms="config.pollIntervalMs"
          @cancel="cancel"
        />
        <ScanResult
          v-else-if="phase === 'done' && result"
          key="result"
          :response="result"
          :preview-url="previewUrl"
          @pick="openPicker('camera')"
        />
        <MessageCard
          v-else-if="phase === 'empty'"
          key="empty"
          kind="empty"
          title="Каталог пока пуст"
          message="В базе ещё нет эталонных фотографий — сравнивать не с чем. Когда каталог проиндексируют, поиск заработает."
          primary-label="Сфотографировать другое"
          @primary="openPicker('camera')"
        />
        <MessageCard
          v-else-if="phase === 'error' && problem"
          key="error"
          kind="error"
          :title="problem.title"
          :message="problem.message"
          primary-label="Повторить поиск"
          secondary-label="Выбрать другое фото"
          @primary="retry"
          @secondary="openPicker('gallery')"
        />
        <UploadHero v-else key="upload" ref="hero" :error="uploadError" :max-mb="maxMb" @pick="openPicker" />
      </Transition>
    </main>

    <AppFooter />

    <!-- Нижняя панель на телефоне: главное действие в зоне большого пальца. -->
    <div v-if="phase === 'done'" class="action-bar">
      <button class="upload-btn upload-btn--bar" type="button" @click="openPicker('camera')">
        <span class="upload-btn__icon" aria-hidden="true"><CameraIcon /></span>
        <span>Сфотографировать другую</span>
      </button>
    </div>

    <!-- capture="environment": на телефоне сразу открывается основная камера. -->
    <input
      ref="cameraInput"
      class="visually-hidden"
      type="file"
      accept="image/jpeg,image/png,image/webp"
      capture="environment"
      tabindex="-1"
      aria-hidden="true"
      @change="onFileChange"
    >
    <input
      ref="galleryInput"
      class="visually-hidden"
      type="file"
      accept="image/jpeg,image/png,image/webp"
      tabindex="-1"
      aria-hidden="true"
      @change="onFileChange"
    >
    <DropOverlay :visible="isDragging" />
  </div>
</template>
