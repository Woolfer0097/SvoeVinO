<script setup lang="ts">
import type { PhotoSource } from "~/utils/photo";

defineProps<{ error: string | null; maxMb: string }>();
const emit = defineEmits<{ pick: [source: PhotoSource] }>();

const button = ref<HTMLButtonElement>();
defineExpose({ focus: () => button.value?.focus() });
</script>

<template>
  <section class="view" aria-labelledby="heroTitle">
    <div class="hero">
      <svg class="hero-art hero-art--a" viewBox="0 0 200 200" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round">
        <path d="M103 10c3 18 2 38 0 60" />
        <path d="M104 38c20-24 58-26 72 4-18 3-42 6-72-4z" />
        <path d="M108 40c18-2 38-3 56 0" />
        <path d="M102 28c-10-10-24-12-30-4-5 7 3 14 9 9" />
        <circle cx="55" cy="88" r="15" /><circle cx="87" cy="88" r="15" /><circle cx="119" cy="88" r="15" /><circle cx="151" cy="88" r="15" />
        <circle cx="71" cy="116" r="15" /><circle cx="103" cy="116" r="15" /><circle cx="135" cy="116" r="15" />
        <circle cx="87" cy="144" r="15" /><circle cx="119" cy="144" r="15" />
        <circle cx="103" cy="172" r="15" />
      </svg>
      <svg class="hero-art hero-art--b" viewBox="0 0 200 200" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round">
        <path d="M103 10c3 18 2 38 0 60" />
        <path d="M104 38c20-24 58-26 72 4-18 3-42 6-72-4z" />
        <circle cx="71" cy="88" r="15" /><circle cx="103" cy="88" r="15" /><circle cx="135" cy="88" r="15" />
        <circle cx="87" cy="116" r="15" /><circle cx="119" cy="116" r="15" />
        <circle cx="103" cy="144" r="15" />
      </svg>

      <p class="eyebrow">Прототип · хакатон «Сканер российских вин»</p>
      <h1 id="heroTitle">Узнайте вино <em>по&nbsp;одной фотографии</em></h1>
      <p class="lead">
        Сфотографируйте бутылку или этикетку — сервис сравнит снимок с&nbsp;эталонными фото каталога
        и&nbsp;покажет самые похожие вина.
      </p>

      <div class="hero-actions">
        <!-- На телефоне кнопка открывает камеру, на компьютере — выбор файла. -->
        <button ref="button" class="upload-btn" type="button" @click="emit('pick', 'camera')">
          <span class="upload-btn__icon" aria-hidden="true"><CameraIcon /></span>
          <span class="only-touch">Сфотографировать бутылку</span>
          <span class="only-pointer">Загрузить фото бутылки</span>
        </button>
        <button class="text-btn only-touch" type="button" @click="emit('pick', 'gallery')">
          или выбрать фото из галереи
        </button>
      </div>

      <p class="hint">
        JPG, PNG или WebP до&nbsp;{{ maxMb }}&nbsp;МБ<span class="only-pointer"> · можно перетащить фото на&nbsp;страницу или вставить из&nbsp;буфера</span>
      </p>
      <p v-if="error" class="inline-error" role="alert">{{ error }}</p>
    </div>

    <ol class="steps">
      <li><span class="step-num">1</span><strong>Снимите этикетку</strong><span>Крупно, ровно и без бликов</span></li>
      <li><span class="step-num">2</span><strong>Сравним с каталогом</strong><span>По эталонным фотографиям вин</span></li>
      <li><span class="step-num">3</span><strong>Выберите своё</strong><span>До 20 похожих вариантов</span></li>
    </ol>
  </section>
</template>
