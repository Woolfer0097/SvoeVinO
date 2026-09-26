# Wine Label Preprocessing Design

## Purpose

Build an independent, CPU-first Python package for preparing wine-bottle photographs for OCR and image retrieval. The package must preserve label detail, make every destructive operation optional, and expose enough metadata to measure whether cylindrical unwrapping helps. It does not train or pretend to provide a bottle/label detector.

The package lives at `modules/wine_label_preprocessing` and does not import or modify `modules/dinov2_retrieval`. Downstream callers may pass its RGB outputs to DINOv2, SigLIP2, OCR, or another model's native processor.

There are no real wine photographs, annotations, OCR engines, retrieval galleries, or DewarpNet checkpoints in the repository. Therefore the delivered evidence is limited to synthetic geometry checks and measured local preprocessing timings. The package must not claim an OCR or retrieval-quality improvement without external data and models.

## Scope

The package provides:

- EXIF-aware loading and explicit RGB/BGR NumPy input handling;
- clamped label/bottle cropping;
- optional resolution limiting, mild luminance processing, CLAHE, and weak unsharp masking;
- a conservative cylindrical inverse map implemented with NumPy and `cv2.remap`;
- explicit cylinder geometry and an optional robust estimate from a bottle mask;
- an optional, lazy adapter for the official DewarpNet implementation and checkpoints;
- a Python API, file/folder CLI, visual comparison sheets, JSON reports, benchmark hooks, and metric functions;
- synthetic and boundary-condition tests.

The package does not provide a detector, segmentation model, OCR model, embedding model, web UI, service, training pipeline, super-resolution, generative inpainting, automatic binarization, or strong denoising.

## Runtime and Packaging

- Python: `>=3.11`.
- Required runtime dependencies: Pillow, NumPy, and `opencv-python-headless`.
- Test dependency: pytest.
- PyTorch and torchvision are available only through a `dewarpnet` optional extra.
- OCR and embedding packages are supplied by downstream applications, not declared by this package.
- The command-line entry point is `wine-label-preprocess`.

The package is independently installable from `modules/wine_label_preprocessing` and has its own `pyproject.toml`, README, source tree, and tests.

## Coordinate and Image Contract

The canonical internal and output representation is a contiguous `np.ndarray` with shape `(height, width, 3)`, dtype `uint8`, and RGB channel order.

Accepted image inputs are:

- `str` or `Path`: decoded with Pillow, transposed with `ImageOps.exif_transpose`, fully loaded, and converted to RGB;
- `PIL.Image.Image`: EXIF-transposed when metadata is present, then converted to RGB;
- NumPy array: accepted only with an explicit `RGB` or `BGR` input color-order value; three-channel `uint8` is required.

All public coordinates refer to the EXIF-oriented full image, never the encoded pre-orientation raster. A bounding box is integer `(x_min, y_min, x_max, y_max)` in pixel coordinates with an inclusive top/left and exclusive bottom/right. Bounds are clamped to the image. A box that is empty after clamping is invalid.

`label_bbox` has priority for the returned crop. If absent, `bottle_bbox` is used. If both are absent, the full image is used. No automatic label or bottle localization is inferred.

A bottle mask is a two-dimensional boolean or `uint8` array. It may match the full oriented image or the supplied bottle bounding box; the latter is placed into full-image coordinates before geometry estimation. Other shapes are rejected. Nonzero mask values are valid foreground.

Cylinder `cx` and `radius` are floating-point pixels in the full oriented image. Angles are radians, increase from left to right, and use the model

```text
x_source = cx + radius * sin(theta)
u_flat = radius * theta
```

`cx` and `radius` describe the visible bottle body, not the center and half-width of the label.

## Public API

The package exposes immutable or effectively immutable dataclasses for configuration and results. The principal call has this shape:

```python
def preprocess(
    image: str | Path | PIL.Image.Image | np.ndarray,
    *,
    annotations: ImageAnnotations | None = None,
    config: PreprocessingConfig | None = None,
    input_color_order: Literal["RGB", "BGR"] | None = None,
    dewarpnet: DewarpNetAdapter | None = None,
) -> PreprocessingResult:
    ...
```

`ImageAnnotations` contains optional `label_bbox`, `bottle_bbox`, `bottle_mask`, and `CylinderGeometry`. `CylinderGeometry` contains `cx`, `radius`, and optional `theta_min`/`theta_max`.

`PreprocessingResult` contains:

- `original_crop`: exact RGB crop at source resolution, with no photometric operation or resize;
- `embedding_image`: resolution-limited branch with embedding photometry disabled by default;
- `ocr_image`: resolution-limited branch using the configured OCR profile;
- `cylindrical_image`: corrected branch or `None`;
- `cylindrical_valid_mask`: boolean mask matching the corrected branch or `None`;
- `dewarpnet_image` and its valid mask when the optional adapter succeeds;
- serializable metadata containing input/output dimensions, coordinates, applied operations, effective parameters, geometry confidence, skip/unavailable reasons, and per-stage milliseconds.

Expected inability to unwrap is represented by `None` plus a stable skip reason, not an exception. Invalid input types, empty boxes, malformed masks, and impossible explicitly supplied values raise a package-specific `PreprocessingError` subclass.

## Baseline Processing

`original_crop` is never resized. Each working branch has a configurable `max_long_side`; values below one are invalid, small images are never enlarged, and aspect ratio is preserved. Ordinary downscaling uses `cv2.INTER_AREA` exactly once.

Photometric operations work only on the L channel of an OpenCV LAB representation and round-trip back to RGB. All are independently configurable:

- brightness correction moves median luminance toward a target with a multiplicative gain clamped to a narrow configured range;
- CLAHE uses configured clip limit and tile-grid size;
- unsharp masking uses a Gaussian blur and a small configured amount.

The default embedding and OCR profiles disable all three operations. A named `ocr_mild` preset enables bounded brightness correction and mild CLAHE; weak unsharp remains separately selectable. The benchmark constructs its corrected variants explicitly rather than changing safe defaults.

No branch performs binarization, strong denoising, generative modification, super-resolution, model normalization, or a model-specific final resize. Those remain the responsibility of the OCR or embedding processor.

## Cylindrical Unwrapping

### Preconditions

The pipeline attempts cylindrical unwrapping only when `label_bbox` is present and one of these sources of geometry is available:

1. valid explicit `cx` and `radius`; or
2. a bottle mask that passes the geometry-confidence checks.

The vertical span comes from `label_bbox`. A bottle bounding box alone does not identify the cylindrical label span and is insufficient.

Explicit angles are used when both limits are supplied. Supplying only one limit is invalid. Otherwise, angles are derived from the label's horizontal bounds:

```text
theta_min = asin((label_x_min - cx) / radius)
theta_max = asin((label_x_max - cx) / radius)
```

The arguments must intersect the visible cylinder domain. Geometry with nonpositive radius, reversed angles, a center/radius inconsistent with the label, or an output below the configured minimum size is rejected with a reason.

### Stretch and Visibility Safety

For the orthographic model the local horizontal stretch is `1 / cos(theta)`. The requested interval is intersected with:

- the visible front side `(-pi/2, pi/2)` with a numerical margin;
- `[-acos(1 / max_stretch), +acos(1 / max_stretch)]`.

`max_stretch` must be greater than one. Clipping is recorded in metadata. If clipping removes the label interval or leaves too little output, unwrapping is skipped. This deliberately discards unsafe silhouette-edge content instead of fabricating it.

### Geometry Estimation from a Mask

Only rows intersecting `label_bbox` participate. For each usable row, the leftmost and rightmost foreground pixels yield a row center and half-width. Robust medians estimate `cx` and radius. Configuration supplies minimum foreground width, minimum usable-row fraction, maximum normalized center median absolute deviation, and maximum radius coefficient of variation.

The estimator reports all confidence statistics. It rejects sparse masks, rows with disconnected or implausibly narrow support, unstable centers, unstable widths, nonfinite results, and geometry inconsistent with the label bounds. It does not claim to segment a label or bottle.

### Inverse Map and Sampling

The output's unscaled width is `radius * (theta_max - theta_min)` and its unscaled height is the label height. One common scale factor, at most one, enforces `max_long_side`; this preserves a consistent horizontal arc-length and vertical pixel scale.

The inverse map is vectorized with NumPy `float32` arrays. For each output pixel center it computes flat arc position, `theta`, `x_source`, and the corresponding linear `y_source`. Full-image coordinates are translated into crop/source coordinates. The map samples the original RGB pixels directly with one `cv2.remap` call, combining unwrapping and optional downscaling rather than resizing twice.

A valid mask is computed from finite map coordinates, source bounds, the safe angular interval, and—when available—the supplied bottle mask. Border pixels outside that mask are filled with a configurable constant RGB value and marked invalid. The mask itself is remapped with nearest-neighbor semantics.

The standalone geometry function returns a structured result containing image, valid mask, effective angles, scale, map dimensions, and metadata. On unreliable parameters, the high-level pipeline leaves `cylindrical_image` absent and continues with baseline outputs. A lower-level strict option may raise for callers that need parameter debugging.

### Known Limitations

The approximation assumes a vertical bottle axis, a camera near label height, an approximately cylindrical body, and visible non-occluded surface content. It cannot reliably correct strong pitch from above or below, a conical body, unknown vertical perspective, folds, occlusion, motion blur, specular highlights, or content beyond the silhouette. These conditions require external pose/shape information or a learned model; the MVP records a skip instead of guessing when its available geometry is inconsistent.

A four-point homography is not used as a substitute because it corrects a planar projective transform, not cylindrical curvature.

## Optional DewarpNet Adapter

The adapter is configured with an official DewarpNet checkout path, a world-coordinate checkpoint, a backward-map checkpoint, device preference, and normalized-grid alignment mode. Construction performs no heavyweight imports or model loading. The first call:

1. verifies the checkout and checkpoint files;
2. imports torch/torchvision and the official model package without installing it into the core package;
3. creates both official architectures, loads checkpoint state dictionaries, moves the models to the selected device, and switches them to evaluation mode;
4. caches both model instances for all later images.

Inference follows the official preprocessing contract: the RGB crop is resized to `256x256`, converted to the channel order and float NCHW tensor expected by the official code, passed through the world-coordinate network, clamped as in the reference implementation, resized to the backward-map network's `128x128` input, and passed through that network.

The predicted normalized two-channel backward map is resized as a coordinate field to the original crop's output width and height. Its normalized values are converted to source pixel coordinates according to the configured `align_corners` convention. The original-resolution RGB crop is then sampled once with `cv2.remap`; the adapter does not return the model's low-resolution resize as an unwarped result. Out-of-bounds map locations form the invalid mask.

Missing optional packages, missing files, incompatible official code, checkpoint-key mismatches, and inference failures become a structured `unavailable`/`failed` DewarpNet result in the high-level pipeline. They are never mislabeled as a successful dewarp. Direct adapter use may request strict exceptions.

The repository does not redistribute official code or weights and does not download Google Drive assets automatically. The README records the exact official repository URL and the two required checkpoint roles. It also records that the official implementation targets document images, uses `256x256` and `128x128` intermediate resolutions, lists an old SciPy pin for its full environment, and has not been validated on wine bottles. DewarpNet remains experimental variant E, not part of the CPU MVP recommendation.

## Comparison Variants

The comparison runner generates the following from the same decoded RGB input and annotations:

- A: original crop;
- B: original crop with the explicit mild OCR photometric profile;
- C: cylindrical unwrapping;
- D: cylindrical unwrapping followed by the same mild profile;
- E: DewarpNet output when the adapter is available.

Missing C, D, or E remain in the report with a reason and are not silently replaced by A. Variants are saved losslessly as PNG by default.

## Benchmark API and Metrics

The benchmark API accepts optional `OCRBackend` and `EmbeddingBackend` protocols. Exactly one instance of each backend is reused across all variants, and the backend identity is recorded. Model loading therefore occurs outside the per-variant loop.

The benchmark manifest identifies images, annotation paths or inline annotations, `series_id`, split (`tuning` or `test`), optional condition tags, OCR truth (`name`, `producer`, `year`, and full text), and retrieval catalog identifier. Validation rejects a `series_id` appearing in both splits. The catalog item order and content hash are fixed in the report and reused for all variants.

When appropriate truth and backends are available, metrics are:

- character error rate using Levenshtein distance normalized by ground-truth length;
- exact normalized match for name, producer, and year, reported separately and jointly;
- cosine-similarity Recall@1 and Recall@5 against the fixed gallery;
- paired degradation rates relative to A, separately for CER, exact fields, Recall@1, and Recall@5.

Text normalization uses Unicode normalization, case folding, whitespace collapse, and trim; both raw and normalized predictions remain in the report. Empty-ground-truth CER behavior is explicit: zero for empty prediction and one for nonempty prediction.

Metrics are aggregated overall and, when present, for `frontal`, `strong_curvature`, `glare`, `blur`, and `occlusion` tags. No metric section is emitted as measured when its backend or truth is absent; it receives `not_run` and a reason.

## Timing Method

Stage timings use `time.perf_counter_ns`. Decode/EXIF time, baseline operations, geometry estimation, map construction/remap, photometry, optional DewarpNet inference, visualization, and file writing are separate fields.

Benchmark p50/p95 values cover in-memory preprocessing from a decoded RGB input and exclude disk decode, model loading, visualization, and output writing. The runner performs configurable warm-up and measured repetitions and records both counts. Reports include source/output dimensions, Python version, NumPy/OpenCV versions, OS, machine architecture, logical CPU count, and available processor description. No fixed latency is promised.

## CLI

The CLI has two commands:

```text
wine-label-preprocess process INPUT --output-dir OUTPUT [options]
wine-label-preprocess benchmark MANIFEST --output-dir OUTPUT [options]
```

`process` accepts one supported image or recursively processes a directory. A JSONL annotation manifest maps relative image paths to per-image boxes, mask paths, and explicit geometry. Single-file box/geometry flags are also supported. Folder mode without a manifest performs only baseline processing and records that geometry was unavailable.

For each input the command writes variant PNG files, valid masks, a labeled contact sheet, and per-image JSON. It also writes a deterministic aggregate `report.json`. Output paths preserve relative input paths to avoid filename collisions.

`benchmark` validates split/series rules, produces A–E, performs configured warmups/repetitions, invokes injected Python backends when the API is used, and writes aggregate timings and available metrics. The packaged CLI has no built-in OCR or embedding implementation, so its metric sections are `not_run`; README examples show how an application supplies backend objects through the Python API.

All CLI failures use a nonzero exit status and a concise stderr message. Per-image expected geometry skips do not abort a folder run.

## Visual Comparison and Reports

Contact sheets use OpenCV only. Each tile preserves aspect ratio on a common canvas and is labeled A–E; unavailable variants show their reason rather than a fabricated image. The sheet is diagnostic and is not fed to models.

JSON serialization converts NumPy scalars and paths to standard JSON values. Every report has a schema version, package version, effective configuration, input identity, annotations, operations, variants, skip/failure reasons, timings, environment, and optional metrics. Image arrays are saved separately and never embedded in JSON.

## Test Strategy

Tests use generated arrays and temporary files; they do not download models or data. Coverage includes:

- EXIF orientation changes pixel dimensions before bbox interpretation;
- RGB/BGR conversion and rejection of ambiguous NumPy input;
- clamped boxes, empty boxes, and full-image fallback;
- no-upscale aspect-preserving resolution limits;
- photometric defaults leave embedding and OCR pixels unchanged apart from an explicitly configured resolution cap;
- brightness/CLAHE/unsharp options preserve shape, dtype, and channel contract;
- analytic map coordinates at the cylinder center and known angles;
- consistent horizontal/vertical scale and expected output dimensions;
- valid-mask behavior and constant border fill;
- stretch clipping near silhouette edges;
- synthetic grid/text rendered onto a cylindrical projection and recovered within a numerical interpolation tolerance;
- stable-mask geometry estimation and rejection of sparse, asymmetric, or unstable masks;
- safe pipeline fallback with stable skip reasons;
- optional DewarpNet dependency/checkpoint failures are explicit and never return a resize;
- comparison A–E naming and missing-variant reporting;
- CER, exact-field, Recall@1/5, degradation-rate, condition grouping, and split leakage checks;
- file/folder CLI outputs, contact sheet, and JSON schema basics;
- timing summaries after warmup.

The end-to-end test suite must run without PyTorch, model downloads, network access, or real photographs.

## Acceptance Criteria

The implementation is complete when:

1. the independent package installs with only its core dependencies and its full test suite passes;
2. documented APIs return the required branches and metadata using the coordinate/color contracts above;
3. cylindrical correction uses a vectorized inverse map and a single final remap, with confidence checks and safe fallback;
4. file and folder CLI runs create variants, visual comparisons, and machine-readable reports;
5. the benchmark can compute all requested metrics when compatible backends/truth are supplied and otherwise reports exactly what was not run;
6. DewarpNet remains optional, lazy, honest about missing prerequisites, and capable of original-resolution final sampling when official code and checkpoints are supplied;
7. documentation clearly separates synthetic correctness and measured preprocessing latency from unmeasured wine-recognition quality.
