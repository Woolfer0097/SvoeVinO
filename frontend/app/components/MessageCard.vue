<script setup lang="ts">
defineProps<{
  kind: "empty" | "error";
  title: string;
  message: string;
  primaryLabel: string;
  secondaryLabel?: string;
}>();
const emit = defineEmits<{ primary: []; secondary: [] }>();

const primaryButton = ref<HTMLButtonElement>();
onMounted(() => primaryButton.value?.focus());
</script>

<template>
  <section class="view" aria-labelledby="messageTitle">
    <div class="message-card">
      <div class="message-icon" aria-hidden="true">
        <svg v-if="kind === 'error'" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round">
          <circle cx="12" cy="12" r="9" />
          <path d="M12 7.5v5.5M12 16.5v.01" />
        </svg>
        <svg v-else viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
          <path d="M7 3h10l-.9 6.2A4.2 4.2 0 0 1 12 13a4.2 4.2 0 0 1-4.1-3.8z" />
          <path d="M12 13v8M8.5 21h7" />
        </svg>
      </div>
      <h2 id="messageTitle">{{ title }}</h2>
      <p>{{ message }}</p>
      <button ref="primaryButton" class="upload-btn upload-btn--small" type="button" @click="emit('primary')">
        <span class="upload-btn__icon" aria-hidden="true"><CameraIcon /></span>
        <span>{{ primaryLabel }}</span>
      </button>
      <button v-if="secondaryLabel" class="text-btn" type="button" @click="emit('secondary')">
        {{ secondaryLabel }}
      </button>
    </div>
  </section>
</template>
