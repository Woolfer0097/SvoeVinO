<script setup lang="ts">
const emit = defineEmits<{ close: []; capture: [file: File]; fallback: [] }>();
const dialog = ref<HTMLDialogElement>();
const video = ref<HTMLVideoElement>();
const error = ref("");
const ready = ref(false);
const busy = ref(false);
let stream: MediaStream | null = null;
let disposed = false;
function stop() { stream?.getTracks().forEach((track) => track.stop()); stream = null; }
function close() { stop(); emit("close"); }
async function takePhoto() {
  if (!video.value?.videoWidth || busy.value) return;
  busy.value = true;
  const canvas = document.createElement("canvas");
  const scale = Math.min(1, 1920 / Math.max(video.value.videoWidth, video.value.videoHeight));
  canvas.width = Math.round(video.value.videoWidth * scale);
  canvas.height = Math.round(video.value.videoHeight * scale);
  const context = canvas.getContext("2d");
  if (!context) { busy.value = false; error.value = "Не удалось сделать снимок."; return; }
  context.drawImage(video.value, 0, 0, canvas.width, canvas.height);
  canvas.toBlob((blob) => {
    busy.value = false;
    if (disposed) return;
    if (!blob) { error.value = "Не удалось сохранить снимок."; return; }
    stop();
    emit("capture", new File([blob], "wine-label.jpg", { type: "image/jpeg" }));
  }, "image/jpeg", .92);
}
onMounted(async () => {
  dialog.value?.showModal();
  try {
    const acquired = await navigator.mediaDevices.getUserMedia({
      audio: false, video: { facingMode: { ideal: "environment" }, width: { ideal: 1920 } },
    });
    if (disposed) { acquired.getTracks().forEach((track) => track.stop()); return; }
    stream = acquired;
    if (video.value) { video.value.srcObject = acquired; await video.value.play(); }
  } catch {
    stop();
    error.value = "Камера недоступна. Разрешите доступ в браузере или выберите готовое фото.";
  }
});
onBeforeUnmount(() => { disposed = true; stop(); dialog.value?.close(); });
</script>
<template>
  <dialog ref="dialog" class="camera-dialog" aria-labelledby="cameraTitle" @cancel.prevent="close">
    <div class="camera-heading"><h2 id="cameraTitle">Этикетка в кадре</h2><button type="button" class="camera-close" aria-label="Закрыть камеру" @click="close">×</button></div>
    <div class="camera-preview">
      <video ref="video" autoplay muted playsinline @loadeddata="ready = true" />
      <p v-if="error" class="camera-error" role="alert">{{ error }}</p>
    </div>
    <p class="camera-tip">Снимите этикетку крупно. Постарайтесь захватить год.</p>
    <button class="upload-btn" type="button" :disabled="!ready || busy || !!error" @click="takePhoto">Сделать снимок</button>
    <button class="text-btn" type="button" @click="stop(); emit('fallback')">Выбрать готовое фото</button>
  </dialog>
</template>
