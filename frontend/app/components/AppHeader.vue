<script setup lang="ts">
defineProps<{ demo: boolean }>();
</script>

<template>
  <header class="site-header">
    <NuxtLink class="brand" to="/" aria-label="SvoeVinO — на главную">
      <svg class="brand-mark" viewBox="0 0 32 32" aria-hidden="true">
        <circle cx="16" cy="16" r="15" fill="none" stroke="currentColor" stroke-width="1.2" opacity=".35" />
        <path d="M11 7h10l-.7 6A4.3 4.3 0 0 1 16 17a4.3 4.3 0 0 1-4.3-4z" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round" />
        <path d="M11.6 11.5h8.8l-.3 1.9A4.1 4.1 0 0 1 16 16.8a4.1 4.1 0 0 1-4.1-3.4z" fill="currentColor" />
        <path d="M16 17v7M12.5 24.5h7" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" />
      </svg>
      <span class="brand-name">Svoe<em>Vin</em>O</span>
    </NuxtLink>
    <div class="header-meta">
      <span class="header-tagline">сканер российских вин</span>
      <span v-if="demo" class="demo-badge">Демо-данные</span>
    </div>
  </header>
</template>
