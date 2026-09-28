/** Откуда брать фото: камера (на телефоне открывается сразу) или галерея. */
export type PhotoSource = "camera" | "gallery";

const ALLOWED_TYPES = ["image/jpeg", "image/jpg", "image/png", "image/webp"];
const ALLOWED_EXTENSIONS = /\.(jpe?g|png|webp)$/i;

/** Проверка до отправки — те же ограничения, что у бэкенда. */
export function validatePhoto(file: File, maxBytes: number): string | null {
  const typeOk = file.type
    ? ALLOWED_TYPES.includes(file.type.toLowerCase())
    : ALLOWED_EXTENSIONS.test(file.name);
  if (!typeOk) return "Этот формат не подходит — загрузите фото в JPG, PNG или WebP.";
  if (file.size === 0) return "Файл пустой — выберите другое фото.";
  if (file.size > maxBytes) {
    return `Фото весит ${formatMb(file.size)} МБ, а можно не больше ${formatMb(maxBytes)} МБ.`;
  }
  return null;
}

/** Сгенерированное «фото» бутылки на полке — для запуска демо по ?demo=1. */
export function makeDemoPhoto(): Promise<File> {
  return new Promise((resolve, reject) => {
    const canvas = document.createElement("canvas");
    canvas.width = 600;
    canvas.height = 800;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      reject(new Error("Canvas is not available"));
      return;
    }

    const background = ctx.createLinearGradient(0, 0, 0, 800);
    background.addColorStop(0, "#DCCAB6");
    background.addColorStop(1, "#8C6F57");
    ctx.fillStyle = background;
    ctx.fillRect(0, 0, 600, 800);
    ctx.fillStyle = "rgba(60, 40, 30, 0.35)";
    ctx.fillRect(0, 700, 600, 100);

    ctx.save();
    ctx.translate(30, 40);
    ctx.scale(2.7, 2.7);
    ctx.fillStyle = "#2B2023";
    ctx.fill(new Path2D(BOTTLE_PATH));
    ctx.fillStyle = "#6E1530";
    ctx.fillRect(86, 10, 28, 40);
    ctx.fillStyle = "#F6EFE3";
    ctx.fillRect(72, 138, 56, 74);
    ctx.fillStyle = "#6E1530";
    ctx.font = "700 23px Georgia, serif";
    ctx.textAlign = "center";
    ctx.fillText("З", 100, 187);
    ctx.restore();

    canvas.toBlob(
      (blob) => (blob ? resolve(new File([blob], "demo-bottle.jpg", { type: "image/jpeg" })) : reject(new Error("Cannot encode demo photo"))),
      "image/jpeg",
      0.9,
    );
  });
}
