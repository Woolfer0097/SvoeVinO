<script setup lang="ts">
defineProps<{ visible: boolean }>();
</script>

<template>
  <div class="drop-overlay" :class="{ 'is-visible': visible }" aria-hidden="true">
    <span>Отпустите фото — начнём поиск</span>
  </div>
</template>
