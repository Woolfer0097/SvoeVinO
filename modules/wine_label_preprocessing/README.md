# Wine Label Preprocessing

Независимый CPU-first модуль предобработки фотографий винных бутылок. Он сохраняет исходный crop, создаёт безопасные ветви для embedding и OCR, умеет разворачивать видимую часть цилиндрической этикетки и формировать сравнение A–E с параметрами и таймингами.

Модуль не содержит детектор бутылки/этикетки, OCR, embedding-модель или обучающие данные. Без внешнего bbox/маски он выполняет только базовую обработку и не изображает надёжную автоматическую локализацию.

## Установка

Требуется Python 3.11 или новее.

```bash
cd modules/wine_label_preprocessing
python3.12 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest -q
```

Обязательные зависимости — NumPy, Pillow и `opencv-python-headless`. PyTorch и torchvision входят только в опциональную группу `dewarpnet`.

## Быстрый запуск

Один файл без разметки — будут созданы безопасные A/B, embedding/OCR ветви и явные причины пропуска C–E:

```bash
.venv/bin/python -m wine_label_preprocessing process photo.jpg \
  --output-dir output
```

Файл с явной геометрией. Углы задаются в радианах:

```bash
.venv/bin/python -m wine_label_preprocessing process photo.jpg \
  --output-dir output \
  --label-bbox 420 610 1510 1420 \
  --cx 965 --radius 690 \
  --theta-min -0.82 --theta-max 0.79 \
  --ocr-mild
```

Папка обрабатывается рекурсивно. Без JSONL-разметки геометрическая ветвь безопасно пропускается:

```bash
.venv/bin/python -m wine_label_preprocessing process ./photos \
  --output-dir ./preprocessed
```

## Контракт изображения и координат

- Все возвращаемые массивы: contiguous `np.ndarray`, `uint8`, `(H, W, 3)`, RGB.
- Для NumPy-входа обязательно явно передать `input_color_order="RGB"` или `"BGR"`.
- Для пути/Pillow сначала применяется `ImageOps.exif_transpose`, затем RGB-конвертация.
- Bbox: `(x_min, y_min, x_max, y_max)`, верхняя/левая границы включены, нижняя/правая исключены.
- Координаты относятся к полному изображению после EXIF-ориентации и ограничиваются его границами.
- `cx` и `radius` описывают корпус бутылки в пикселях, а не центр и половину ширины этикетки.
- Маска бутылки может иметь размер полного ориентированного изображения либо `bottle_bbox`.

## Python API

```python
from pathlib import Path

from wine_label_preprocessing import (
    BBox,
    CylinderGeometry,
    ImageAnnotations,
    PreprocessingConfig,
    preprocess,
)

result = preprocess(
    Path("photo.jpg"),
    annotations=ImageAnnotations(
        label_bbox=BBox(420, 610, 1510, 1420),
        cylinder=CylinderGeometry(
            cx=965.0,
            radius=690.0,
            theta_min=-0.82,
            theta_max=0.79,
        ),
    ),
    config=PreprocessingConfig(),
)

original_rgb = result.original_crop
embedding_rgb = result.embedding_image
ocr_rgb = result.ocr_image
cylindrical_rgb = result.cylindrical_image
valid_mask = result.cylindrical_valid_mask
metadata = result.metadata
```

`original_crop` не ресайзится и не получает фотометрических изменений. Ограничение длинной стороны рабочих ветвей настраивается; маленькие изображения не увеличиваются. Финальные resize и нормализацию DINOv2/SigLIP2 должен выполнять штатный processor модели.

Фотометрия работает только по L-каналу LAB. В embedding и OCR ветвях яркость, CLAHE и unsharp по умолчанию выключены. CLI-флаг `--ocr-mild` включает ограниченную коррекцию яркости и мягкий CLAHE; `--ocr-unsharp` отдельно включает слабый unsharp.

## Цилиндрическая развёртка

Используется ортографическое приближение:

```text
x_source = cx + radius * sin(theta)
u_flat = radius * theta
```

Для каждого пикселя результата строится обратная карта `float32`, после чего исходное RGB-изображение семплируется одним `cv2.remap`. Если нужен downscale, он объединяется с этой же картой. Горизонтальный масштаб по длине дуги и вертикальный масштаб согласованы.

Диапазон ограничивается видимой передней стороной и `max_stretch`, где локальное растяжение равно `1/cos(theta)`. Возвращается valid mask. При невалидной геометрии, слабой/нестабильной маске или слишком маленьком результате развёртка пропускается, а базовые ветви остаются доступны.

Оценка `cx/radius` по маске использует левые/правые границы корпуса на строках этикетки и проверяет покрытие, связность, стабильность центра и ширины. Это не сегментатор.

Ограничения MVP: сильный наклон камеры сверху/снизу, конический корпус, неизвестная перспектива, перекрытия, блики, складки и размытие. Homography не используется как замена: четыре точки исправляют плоскую перспективу, но не кривизну цилиндра.

## JSONL-разметка папки

```json
{"image":"batch/front.jpg","annotations":{"label_bbox":[420,610,1510,1420],"bottle_bbox":[250,100,1700,2100],"bottle_mask":"masks/front.png","cylinder":{"cx":965,"radius":690,"theta_min":-0.82,"theta_max":0.79}}}
```

Пути масок разрешаются относительно файла манифеста. Для папки:

```bash
.venv/bin/python -m wine_label_preprocessing process ./photos \
  --annotations ./annotations.jsonl \
  --output-dir ./preprocessed
```

## Сравнение A–E и benchmark

- A — исходный crop;
- B — исходный crop с mild-фотометрией;
- C — цилиндрическая развёртка;
- D — развёртка с той же mild-фотометрией;
- E — DewarpNet, только если адаптер доступен.

Benchmark-манифест:

```json
{"sample_id":"query-001","image":"photos/query-001.jpg","series_id":"shoot-17","split":"test","conditions":["glare","strong_curvature"],"catalog_id":"wine-42","truth":{"full_text":"Example Winery Reserve 2020","name":"Reserve","producer":"Example Winery","year":"2020"},"annotations":{"label_bbox":[420,610,1510,1420],"cylinder":{"cx":965,"radius":690}}}
```

```bash
.venv/bin/python -m wine_label_preprocessing benchmark benchmark.jsonl \
  --output-dir benchmark-output \
  --warmup-runs 2 --measured-runs 20
```

CLI не содержит OCR/embedding-модель, поэтому их секции честно получают `not_run`. Python API `run_benchmark(...)` принимает по одному экземпляру `OCRBackend` и `EmbeddingBackend` и переиспользует их для всех вариантов. При наличии истины/фиксированной галереи считаются CER, exact name/producer/year, Recall@1/5 и paired degradation относительно A. Одинаковый `series_id` запрещён одновременно в `tuning` и `test`; фиксированная галерея записывается с SHA-256.

Тайминги p50/p95 измеряют in-memory preprocessing после указанного прогрева. Decode, создание модели, визуализация и запись файлов вынесены из измерения. В отчёте фиксируются размеры, версии Python/NumPy/OpenCV, платформа и CPU-информация. Фиксированная задержка не обещается.

## Синтетический стенд

Он проверяет только корректность геометрии на сетке и тексте:

```bash
.venv/bin/python examples/generate_synthetic_case.py /tmp/wine-preprocessing-demo
.venv/bin/python -m wine_label_preprocessing benchmark \
  /tmp/wine-preprocessing-demo/benchmark.jsonl \
  --output-dir /tmp/wine-preprocessing-result \
  --warmup-runs 2 --measured-runs 20
```

Откройте `/tmp/wine-preprocessing-result/synthetic-bottle/comparison.png`; параметры и p50/p95 находятся в `report.json`.

## DewarpNet (эксперимент E)

Основа: [официальный DewarpNet](https://github.com/cvlab-stonybrook/DewarpNet) и [статья ICCV 2019](https://www3.cs.stonybrook.edu/~cvl/content/papers/2019/SagnikKe_ICCV19.pdf). Модель обучена на документах Doc3D и не считается автоматически подходящей для бутылок. Статья о [3D viewpoint augmentation винных этикеток](https://arxiv.org/abs/2404.08820) подтверждает полезность моделирования ракурса для обучения, но не доказывает пользу document dewarping на фото бутылок.

Официальные веса отсутствуют в этом репозитории и опубликованы авторами через Google Drive. Нужны два файла: world-coordinate checkpoint (`unetnc`) и backward-map checkpoint (`dnetccnl`), а также checkout официального репозитория.

```bash
.venv/bin/pip install -e '.[dewarpnet]'
.venv/bin/python -m wine_label_preprocessing process photo.jpg \
  --output-dir output \
  --dewarpnet-repo /path/to/DewarpNet \
  --dewarpnet-wc-checkpoint /path/to/unetnc_doc3d.pkl \
  --dewarpnet-bm-checkpoint /path/to/dnetccnl_doc3d.pkl \
  --dewarpnet-device cpu
```

Адаптер лениво и один раз загружает обе сети. Вход world-coordinate сети — `256×256`, backward-map сети — `128×128`. Предсказанная нормализованная карта переносится на размер исходного crop, преобразуется в пиксельные координаты и семплирует исходный crop один раз. Если checkout, зависимости или checkpoint несовместимы, E получает `unavailable`/`failed`; resize никогда не выдаётся за результат DewarpNet.

Официальный `requirements.txt` содержит устаревший `scipy==1.1.0`, а код и веса ориентированы на воспроизведение document benchmark. Полную совместимость с современным PyTorch и качество на бутылках можно подтвердить только после получения официальных checkpoint и реальных фотографий.

## Что пока не измерено

В репозитории нет реальных фото, OCR backend, каталожной галереи, разметки полей и checkpoint DewarpNet. Поэтому нельзя утверждать улучшение CER, распознавания названия/производителя/года или Recall@1/5. Текущие тесты доказывают контракт, безопасные fallback, корректность синтетической геометрии и работу отчётности; продуктовое качество требует отдельного tuning/test набора без пересечения серий съёмки.
