# SvoeVinO

Распознавание вина по фотографии: preprocessing → DINOv2 giant → SuperPoint/LightGlue,
параллельно OCR → multilingual E5, затем объединение результатов.
Пропущенные DINO OCR-кандидаты проходят дополнительную геометрическую проверку.
Веса visual/OCR 0.5/0.5, палитра фото — слабый 5% сигнал; явно прочитанный год
сверяется с каталогом. Score не является вероятностью правильного ответа.

Минималистичный [фронтенд](frontend/README.md): http://localhost:3000 — камера,
галерея, одна карточка с реальными данными, web-фото и ссылкой на Вино своё.

Общий backend: [запуск, GPU и два endpoint](modules/wine_pipeline/README.md).
Организаторский API: `POST /v1/eval/predict`, multipart `image`, ответ `{"slug":"..."}`.
Асинхронный режим: `?wait=false` и `GET /image/status?job_id=...`.

[Локальный Docker pipeline](compose.pipeline.yml) использует GPU для DINO/SuperPoint;
[продовый GPU override](compose.pipeline.gpu.yml) также включает GPU для OCR/E5.
