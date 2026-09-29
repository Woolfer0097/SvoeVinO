<script setup lang="ts">
import grape from "../../bg_element.png";
const elements = ref<{ left: string; top: string; transform: string; width: string }[]>([]);
onMounted(() => {
  elements.value = [0, 1, 2].map(index => ({
    left: index === 1 ? `${83 + Math.random() * 8}%` : `${-13 + Math.random() * 10}%`,
    top: `${index * 31 + 3 + Math.random() * 10}%`,
    transform: `rotate(${-35 + Math.random() * 70}deg) scaleX(${index === 1 ? -1 : 1})`,
    width: `${230 + Math.random() * 100}px`,
  }));
});
</script>
<template>
  <div class="background-decoration" aria-hidden="true">
    <img v-for="(element, index) in elements" :key="index" :src="grape" :style="element" alt="" draggable="false">
  </div>
</template>
<style scoped>
.background-decoration { position: fixed; inset: 0; overflow: hidden; z-index: 0; pointer-events: none; user-select: none; }
.background-decoration img { position: absolute; max-width: none; opacity: .075; mix-blend-mode: multiply; }
@media (max-width: 600px) { .background-decoration img { width: 185px !important; opacity: .045; } }
</style>
