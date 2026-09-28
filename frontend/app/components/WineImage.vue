<script setup lang="ts">
import type { WineCandidate } from "#shared/types/recognition";

const props = defineProps<{ candidate: WineCandidate; name: string }>();

const config = useRuntimeConfig().public;
const failed = ref(false);

// Встроенный мок фото эталонов не отдаёт; настоящий бэкенд — через GET /images.
const src = computed(() => {
  if (config.demoData || failed.value || !props.candidate.best_image_uri) return null;
  const uri = encodeURIComponent(props.candidate.best_image_uri);
  return `${config.apiBase}/images?uri=${uri}&max_side=600`;
});

watch(() => props.candidate, () => (failed.value = false));
</script>

<template>
  <img v-if="src" :src="src" :alt="name" loading="lazy" decoding="async" @error="failed = true">
  <span v-else class="art">
    <BottleArt :wine="{ ...candidate, name }" :label="`Иллюстрация: ${name}`" />
  </span>
</template>
