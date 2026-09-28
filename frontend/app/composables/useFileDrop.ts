/** Перетаскивание файла на страницу и вставка из буфера обмена. */
export function useFileDrop(onFile: (file: File) => void, isEnabled: () => boolean) {
  const isDragging = ref(false);
  let depth = 0;

  const hasFiles = (event: DragEvent) => Array.from(event.dataTransfer?.types ?? []).includes("Files");

  function onDragEnter(event: DragEvent) {
    if (!hasFiles(event) || !isEnabled()) return;
    event.preventDefault();
    depth += 1;
    isDragging.value = true;
  }

  function onDragOver(event: DragEvent) {
    if (!hasFiles(event)) return;
    event.preventDefault();
    if (event.dataTransfer) event.dataTransfer.dropEffect = isEnabled() ? "copy" : "none";
  }

  function onDragLeave(event: DragEvent) {
    if (!hasFiles(event)) return;
    depth = Math.max(0, depth - 1);
    if (depth === 0) isDragging.value = false;
  }

  function onDrop(event: DragEvent) {
    if (!hasFiles(event)) return;
    event.preventDefault();
    depth = 0;
    isDragging.value = false;
    const file = event.dataTransfer?.files[0];
    if (file && isEnabled()) onFile(file);
  }

  function onPaste(event: ClipboardEvent) {
    const file = event.clipboardData?.files[0];
    if (!file || !isEnabled()) return;
    event.preventDefault();
    onFile(file);
  }

  onMounted(() => {
    document.addEventListener("dragenter", onDragEnter);
    document.addEventListener("dragover", onDragOver);
    document.addEventListener("dragleave", onDragLeave);
    document.addEventListener("drop", onDrop);
    document.addEventListener("paste", onPaste);
  });

  onBeforeUnmount(() => {
    document.removeEventListener("dragenter", onDragEnter);
    document.removeEventListener("dragover", onDragOver);
    document.removeEventListener("dragleave", onDragLeave);
    document.removeEventListener("drop", onDrop);
    document.removeEventListener("paste", onPaste);
  });

  return { isDragging };
}
