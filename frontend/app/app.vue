<template>
  <NuxtRouteAnnouncer />
  <div class="app-shell">
    <BackgroundDecoration />
    <div class="app-content"><NuxtPage /></div>
  </div>
</template>
<style>
.app-shell { position: relative; isolation: isolate; }
.app-content { position: relative; z-index: 1; }
</style>
