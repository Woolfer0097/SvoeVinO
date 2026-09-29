<script setup lang="ts">
import type { PhotoSource } from "~/utils/photo";
defineProps<{ error: string | null; maxMb: string; ready: boolean }>();
const emit = defineEmits<{ pick: [source: PhotoSource] }>();
const button = ref<HTMLButtonElement>();
defineExpose({ focus: () => button.value?.focus() });
</script>
<template>
  <section class="view upload-panel" aria-labelledby="heroTitle">
    <div class="upload-mark" aria-hidden="true"><CameraIcon /></div>
    <p class="eyebrow">Сканер винных этикеток</p>
    <h1 id="heroTitle">Что за <em>вино?</em></h1>
    <p class="upload-lead">Сфотографируйте этикетку.<br>Найдём вино и откроем его историю.</p>
    <div class="upload-actions">
      <button ref="button" class="upload-btn" type="button" :disabled="!ready" @click="emit('pick', 'camera')">
        <span class="upload-btn__icon" aria-hidden="true"><CameraIcon /></span><span>Сделать фото</span>
      </button>
      <button class="secondary-btn" type="button" :disabled="!ready" @click="emit('pick', 'gallery')">Выбрать из галереи</button>
    </div>
    <p class="upload-tip">Этикетка крупно, год в кадре, без бликов.</p>
    <p class="hint">JPG, PNG или WebP · до {{ maxMb }} МБ</p>
    <p v-if="error" class="inline-error" role="alert">{{ error }}</p>
  </section>
</template>
