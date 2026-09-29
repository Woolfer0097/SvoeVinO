<script setup lang="ts">
import type { WineCandidate } from "#shared/types/recognition";
const props = defineProps<{ candidate: WineCandidate; name: string }>();
const failed = ref(false);
const src = computed(() => props.candidate.web_photo_uri && !failed.value
  ? `/api/images?uri=${encodeURIComponent(props.candidate.web_photo_uri)}` : null);
watch(() => props.candidate.web_photo_uri, () => (failed.value = false));
</script>
<template>
  <img v-if="src" :src="src" :alt="`Вино ${name} — фото из каталога`" decoding="async" @error="failed = true">
  <div v-else class="photo-placeholder"><CameraIcon /><span>В каталоге пока нет фото</span></div>
</template>
