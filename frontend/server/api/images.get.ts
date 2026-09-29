/** Original file retains transparency; paths are confined to web_photos. */
export default defineEventHandler(async (event) => {
  const uri = getQuery(event).uri;
  if (typeof uri !== "string" || !/^\/data\/web_photos\/[^/\\]+\.(jpg|jpeg|png|webp)$/i.test(uri) || uri.includes(".."))
    throw createError({ statusCode: 400, message: "Некорректный путь фото" });
  const url = new URL("/images", useRuntimeConfig(event).dinoUrl);
  url.searchParams.set("uri", uri);
  url.searchParams.set("max_side", "0");
  let upstream: Response;
  try { upstream = await fetch(url, { signal: AbortSignal.timeout(15_000) }); }
  catch { throw createError({ statusCode: 503, message: "Фото недоступно" }); }
  if (!upstream.ok) throw createError({ statusCode: upstream.status, message: "Фото не найдено" });
  const contentType = upstream.headers.get("content-type") ?? "";
  if (!/^image\/(jpeg|png|webp)/i.test(contentType))
    throw createError({ statusCode: 502, message: "Некорректный формат фото" });
  return new Response(upstream.body, { headers: {
    "Content-Type": contentType, "Cache-Control": "public, max-age=86400",
    "X-Content-Type-Options": "nosniff",
  } });
});
