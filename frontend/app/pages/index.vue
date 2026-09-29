<script setup lang="ts">
/* Единственная страница: фото → настоящий pipeline → одна карточка вина. */
import type { PhotoSource } from "~/utils/photo";

const config = useRuntimeConfig().public;
const recognition = useRecognition();
const { phase, stage, progress, jobId, result, problem, isBusy, canResume } = recognition;

const cameraInput = ref<HTMLInputElement>();
const galleryInput = ref<HTMLInputElement>();
const hero = ref<{ focus: () => void }>();
const uploadError = ref<string | null>(null);
const previewUrl = ref<string | null>(null);
const cameraOpen = ref(false);
const ready = ref(false);

const maxMb = formatMb(config.maxBytes);

function setPreview(file: File) {
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value);
  previewUrl.value = URL.createObjectURL(file);
}

function focusUpload() {
  nextTick(() => hero.value?.focus());
}

function openPicker(source: PhotoSource = "camera") {
  if (source === "camera" && window.isSecureContext && typeof navigator.mediaDevices?.getUserMedia === "function") {
    cameraOpen.value = true;
    return;
  }
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
  recognition.start(file);
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
  recognition.retry();
}

const { isDragging } = useFileDrop(handleFile, () => !isBusy.value);

function onKeydown(event: KeyboardEvent) {
  if (event.key === "Escape" && isBusy.value) cancel();
}

onMounted(() => {
  ready.value = true;
  document.addEventListener("keydown", onKeydown);
});

onBeforeUnmount(() => {
  document.removeEventListener("keydown", onKeydown);
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value);
});
</script>

<template>
  <div class="page" :data-ready="ready" :class="{ 'page--with-bar': phase === 'done' }">
    <AppHeader :demo="false" />

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
        <section v-else-if="phase === 'empty'" key="empty" class="result-view">
          <MessageCard kind="empty" title="Не найдено вариантов в каталоге"
            message="Снимите одну этикетку крупнее, без бликов, и попробуйте ещё раз."
            primary-label="Сфотографировать ещё раз" @primary="openPicker('camera')" />
          <RecognitionFeedback v-if="result" :key="result.request_id" :job-id="result.request_id" unrecognized />
        </section>
        <MessageCard
          v-else-if="phase === 'error' && problem"
          key="error"
          kind="error"
          :title="problem.title"
          :message="problem.message"
          :primary-label="canResume ? 'Проверить результат' : 'Повторить поиск'"
          secondary-label="Выбрать другое фото"
          @primary="retry"
          @secondary="openPicker('gallery')"
        />
        <UploadHero v-else key="upload" ref="hero" :ready="ready" :error="uploadError" :max-mb="maxMb" @pick="openPicker" />
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
      :disabled="!ready"
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
      :disabled="!ready"
      accept="image/jpeg,image/png,image/webp"
      tabindex="-1"
      aria-hidden="true"
      @change="onFileChange"
    >
    <DropOverlay :visible="isDragging" />
    <CameraCapture
      v-if="cameraOpen"
      @close="cameraOpen = false; focusUpload()"
      @capture="cameraOpen = false; handleFile($event)"
      @fallback="cameraOpen = false; openPicker('gallery')"
    />
  </div>
</template>
