# Wine Label Preprocessing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an independently installable CPU-first package that produces safe baseline, OCR, cylindrical-unwrapped, and optional DewarpNet variants with measurable timings and honest comparison reports.

**Architecture:** Keep image contracts, photometry, geometry, optional learned dewarping, benchmarking, and CLI/output concerns in focused modules. All core data is contiguous RGB `uint8`; the high-level pipeline converts once, records every operation, and treats unreliable geometry as an expected skip. OCR and embedding evaluators are injected protocols so the package remains model-independent.

**Tech Stack:** Python 3.11+, NumPy, Pillow, OpenCV headless, pytest; optional PyTorch/torchvision for DewarpNet.

**Spec:** `docs/superpowers/specs/2026-09-22-wine-label-preprocessing-design.md`

**Execution status:** Completed inline on 2026-09-22. Final verification: 62 tests passed, sdist/wheel built, and the synthetic 640x480 benchmark produced all expected non-model artifacts. Documentation remains uncommitted by request.

## Global Constraints

- Implement under `modules/wine_label_preprocessing`; do not modify `modules/dinov2_retrieval`.
- Canonical arrays are contiguous RGB `uint8` with shape `(H, W, 3)`.
- Public coordinates are integer half-open boxes in the EXIF-oriented full image.
- Do not add a detector, automatic binarization, strong denoising, generative processing, super-resolution, a web service, or training infrastructure.
- Keep embedding photometry disabled by default and leave model-specific resize/normalization to the model processor.
- Cylindrical correction must use a vectorized inverse map and one final `cv2.remap`.
- DewarpNet must be optional and lazy; missing code, weights, or dependencies must never yield a fake successful result.
- Do not claim wine-recognition improvements without real photos, annotations, OCR, and retrieval backends.
- Documentation files remain uncommitted per user instruction.

---

### Task 1: Package scaffold, contracts, EXIF loading, crop, and resize

**Files:**
- Create: `modules/wine_label_preprocessing/pyproject.toml`
- Create: `modules/wine_label_preprocessing/src/wine_label_preprocessing/__init__.py`
- Create: `modules/wine_label_preprocessing/src/wine_label_preprocessing/errors.py`
- Create: `modules/wine_label_preprocessing/src/wine_label_preprocessing/models.py`
- Create: `modules/wine_label_preprocessing/src/wine_label_preprocessing/image_io.py`
- Test: `modules/wine_label_preprocessing/tests/test_image_io.py`

**Interfaces:**
- Produces: `BBox`, `CylinderGeometry`, `ImageAnnotations`, `PhotometricConfig`, `UnwrapConfig`, `PreprocessingConfig`, `PreprocessingResult`, `load_rgb`, `crop_rgb`, and `limit_resolution`.
- Consumes: only NumPy, Pillow, OpenCV, and the standard library.

- [ ] **Step 1: Write failing tests for RGB/BGR input, EXIF, bbox clamping, and resize**

```python
def test_numpy_requires_explicit_color_order() -> None:
    with pytest.raises(ImageInputError, match="color order"):
        load_rgb(np.zeros((2, 3, 3), dtype=np.uint8))

def test_bgr_is_converted_to_rgb() -> None:
    bgr = np.array([[[3, 2, 1]]], dtype=np.uint8)
    assert load_rgb(bgr, input_color_order="BGR").array.tolist() == [[[1, 2, 3]]]

def test_bbox_is_clamped_and_half_open() -> None:
    image = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    crop, effective = crop_rgb(image, BBox(-2, 1, 20, 5))
    assert crop.shape == (4, 8, 3)
    assert effective == BBox(0, 1, 8, 5)

def test_limit_resolution_never_upscales() -> None:
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    assert limit_resolution(image, 40).shape == (10, 20, 3)
    assert limit_resolution(image, 10).shape == (5, 10, 3)
```

- [ ] **Step 2: Run the focused tests and verify import failures**

Run: `cd modules/wine_label_preprocessing && python -m pytest tests/test_image_io.py -v`

Expected: FAIL because the package modules do not exist.

- [ ] **Step 3: Add package metadata and typed contracts**

```python
@dataclass(frozen=True, slots=True)
class BBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int

@dataclass(frozen=True, slots=True)
class CylinderGeometry:
    cx: float
    radius: float
    theta_min: float | None = None
    theta_max: float | None = None

@dataclass(slots=True)
class ImageAnnotations:
    label_bbox: BBox | None = None
    bottle_bbox: BBox | None = None
    bottle_mask: np.ndarray | None = None
    cylinder: CylinderGeometry | None = None
```

`pyproject.toml` must declare Pillow, NumPy, OpenCV headless, a `dev` pytest extra, a `dewarpnet` torch/torchvision extra, setuptools package discovery, and the `wine-label-preprocess` entry point.

- [ ] **Step 4: Implement EXIF-aware load, crop validation, and proportional resize**

```python
def load_rgb(image: ImageInput, *, input_color_order: ColorOrder | None = None) -> LoadedImage:
    if isinstance(image, np.ndarray):
        if input_color_order not in ("RGB", "BGR"):
            raise ImageInputError("NumPy input requires explicit RGB or BGR color order")
        validate_rgb_array(image)
        rgb = image if input_color_order == "RGB" else image[..., ::-1]
        return LoadedImage(np.ascontiguousarray(rgb), {"exif_transposed": False})
    if isinstance(image, Image.Image):
        oriented = ImageOps.exif_transpose(image)
        array = np.asarray(oriented.convert("RGB"), dtype=np.uint8)
        return LoadedImage(np.ascontiguousarray(array), {"exif_transposed": oriented.size != image.size})
    try:
        with Image.open(Path(image)) as source:
            original_size = source.size
            oriented = ImageOps.exif_transpose(source)
            array = np.asarray(oriented.convert("RGB"), dtype=np.uint8)
            return LoadedImage(np.ascontiguousarray(array), {"exif_transposed": oriented.size != original_size})
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ImageInputError(f"Cannot decode image: {image}") from exc

def limit_resolution(image: np.ndarray, max_long_side: int | None) -> np.ndarray:
    if max_long_side is None or max(image.shape[:2]) <= max_long_side:
        return image.copy()
    scale = max_long_side / max(image.shape[:2])
    size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)
```

- [ ] **Step 5: Run Task 1 tests**

Run: `cd modules/wine_label_preprocessing && python -m pytest tests/test_image_io.py -v`

Expected: PASS.

- [ ] **Step 6: Record a code checkpoint without committing documentation**

Run: `git diff --check && git status --short`

Expected: new package files and untracked `docs/`; no whitespace errors and no Git commit.

---

### Task 2: Photometry and safe baseline pipeline

**Files:**
- Create: `modules/wine_label_preprocessing/src/wine_label_preprocessing/photometry.py`
- Create: `modules/wine_label_preprocessing/src/wine_label_preprocessing/pipeline.py`
- Modify: `modules/wine_label_preprocessing/src/wine_label_preprocessing/__init__.py`
- Test: `modules/wine_label_preprocessing/tests/test_photometry.py`
- Test: `modules/wine_label_preprocessing/tests/test_pipeline.py`

**Interfaces:**
- Consumes: contracts and I/O helpers from Task 1.
- Produces: `apply_photometric`, `mild_ocr_profile`, and `preprocess` with baseline outputs and metadata; geometry fields remain absent with a stable reason until Task 3.

- [ ] **Step 1: Write failing tests for default identity and opt-in luminance operations**

```python
def test_default_profiles_do_not_change_pixels() -> None:
    image = np.full((20, 30, 3), (30, 80, 150), dtype=np.uint8)
    result = preprocess(image, input_color_order="RGB")
    np.testing.assert_array_equal(result.original_crop, image)
    np.testing.assert_array_equal(result.embedding_image, image)
    np.testing.assert_array_equal(result.ocr_image, image)

def test_photometry_preserves_contract() -> None:
    output, operations = apply_photometric(
        np.full((32, 48, 3), 50, dtype=np.uint8), mild_ocr_profile()
    )
    assert output.shape == (32, 48, 3)
    assert output.dtype == np.uint8
    assert {item["name"] for item in operations} == {"brightness", "clahe"}
```

- [ ] **Step 2: Run focused tests and verify they fail**

Run: `cd modules/wine_label_preprocessing && python -m pytest tests/test_photometry.py tests/test_pipeline.py -v`

Expected: FAIL because photometry and pipeline modules do not exist.

- [ ] **Step 3: Implement bounded LAB luminance processing**

```python
def apply_photometric(image: RGBArray, config: PhotometricConfig) -> tuple[RGBArray, list[dict[str, object]]]:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    luminance = lab[..., 0]
    if config.brightness_enabled:
        median = float(np.median(luminance))
        gain = float(np.clip(config.target_luminance / max(median, 1.0), config.min_gain, config.max_gain))
        luminance = np.clip(luminance.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    if config.clahe_enabled:
        luminance = cv2.createCLAHE(config.clahe_clip_limit, config.clahe_grid_size).apply(luminance)
    lab[..., 0] = luminance
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    if config.unsharp_enabled:
        blur = cv2.GaussianBlur(rgb, (0, 0), config.unsharp_sigma)
        rgb = cv2.addWeighted(rgb, 1.0 + config.unsharp_amount, blur, -config.unsharp_amount, 0)
    return np.ascontiguousarray(rgb), operations
```

- [ ] **Step 4: Implement baseline preprocess and timing metadata**

The pipeline must load once, choose `label_bbox`, then `bottle_bbox`, then full image, preserve an exact `original_crop`, separately resolution-limit embedding/OCR branches, and apply only their configured profiles. Use `perf_counter_ns` and emit milliseconds. Set cylindrical skip reason to `label_bbox_missing` or `geometry_missing` without raising.

- [ ] **Step 5: Run Task 2 tests**

Run: `cd modules/wine_label_preprocessing && python -m pytest tests/test_photometry.py tests/test_pipeline.py -v`

Expected: PASS.

- [ ] **Step 6: Run the accumulated suite and inspect the diff**

Run: `cd modules/wine_label_preprocessing && python -m pytest -q && git diff --check`

Expected: PASS and no whitespace errors.

---

### Task 3: Bottle-mask geometry estimation and cylindrical inverse mapping

**Files:**
- Create: `modules/wine_label_preprocessing/src/wine_label_preprocessing/geometry.py`
- Modify: `modules/wine_label_preprocessing/src/wine_label_preprocessing/pipeline.py`
- Modify: `modules/wine_label_preprocessing/src/wine_label_preprocessing/models.py`
- Modify: `modules/wine_label_preprocessing/src/wine_label_preprocessing/__init__.py`
- Test: `modules/wine_label_preprocessing/tests/test_geometry.py`
- Test: `modules/wine_label_preprocessing/tests/test_pipeline.py`

**Interfaces:**
- Consumes: full RGB image, full-image `label_bbox`, explicit geometry or full-image bottle mask, and `UnwrapConfig`.
- Produces: `GeometryEstimate`, `CylindricalResult`, `estimate_cylinder_from_mask`, `cylindrical_unwrap`, and integrated safe fallback.

- [ ] **Step 1: Write failing analytic, synthetic, mask-confidence, and fallback tests**

```python
def test_unwrap_center_maps_to_cylinder_center() -> None:
    yy, xx = np.mgrid[:40, :120]
    image = np.stack((xx, yy, np.zeros_like(xx)), axis=-1).astype(np.uint8)
    result = cylindrical_unwrap(
        image,
        label_bbox=BBox(10, 5, 110, 35),
        geometry=CylinderGeometry(cx=60.0, radius=50.0, theta_min=-0.4, theta_max=0.4),
    )
    center = result.image[result.image.shape[0] // 2, result.image.shape[1] // 2]
    assert center[0] == pytest.approx(60, abs=1)

def test_mask_estimator_rejects_unstable_widths() -> None:
    mask = np.zeros((80, 120), dtype=np.uint8)
    for y in range(10, 70):
        half_width = 35 if y % 2 else 12
        mask[y, 60 - half_width : 60 + half_width] = 1
    estimate = estimate_cylinder_from_mask(mask, BBox(20, 10, 100, 70), UnwrapConfig())
    assert not estimate.reliable
    assert estimate.reason == "mask_radius_unstable"

def test_pipeline_keeps_baselines_when_geometry_is_unreliable() -> None:
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    sparse_mask = np.zeros((80, 120), dtype=np.uint8)
    sparse_mask[30, 40:80] = 1
    annotations = ImageAnnotations(label_bbox=BBox(30, 20, 90, 60), bottle_mask=sparse_mask)
    result = preprocess(image, annotations=annotations, input_color_order="RGB")
    assert result.cylindrical_image is None
    assert result.metadata["cylindrical"]["status"] == "skipped"
```

- [ ] **Step 2: Run focused geometry tests and verify failures**

Run: `cd modules/wine_label_preprocessing && python -m pytest tests/test_geometry.py tests/test_pipeline.py -v`

Expected: FAIL because geometry functions are missing.

- [ ] **Step 3: Implement robust row-wise mask geometry**

```python
for y in range(label_bbox.y_min, label_bbox.y_max):
    xs = np.flatnonzero(mask[y])
    if xs.size >= config.mask_min_foreground_width:
        centers.append((float(xs[0]) + float(xs[-1])) / 2.0)
        radii.append((float(xs[-1]) - float(xs[0]) + 1.0) / 2.0)
cx = float(np.median(centers))
radius = float(np.median(radii))
center_mad_ratio = median_abs_deviation(centers) / radius
radius_cv = float(np.std(radii) / radius)
```

Also count foreground transitions/runs per row so disconnected masks cannot pass merely because their extremes look wide. Return measured row fraction, MAD ratio, coefficient of variation, and the first stable rejection reason.

- [ ] **Step 4: Implement angle safety and a single vectorized inverse map**

```python
safe_angle = math.acos(1.0 / config.max_stretch)
theta_min = max(requested_min, -safe_angle, -math.pi / 2 + config.angle_margin)
theta_max = min(requested_max, safe_angle, math.pi / 2 - config.angle_margin)
unscaled_width = geometry.radius * (theta_max - theta_min)
scale = min(1.0, config.max_long_side / max(unscaled_width, label_bbox.height))
out_w = max(1, int(round(unscaled_width * scale)))
out_h = max(1, int(round(label_bbox.height * scale)))
flat_u = np.arange(out_w, dtype=np.float32) / np.float32(scale)
theta = np.float32(theta_min) + flat_u / np.float32(geometry.radius)
map_x = geometry.cx + geometry.radius * np.sin(theta)
map_y = label_bbox.y_min + np.arange(out_h, dtype=np.float32) / np.float32(scale)
map_x = np.broadcast_to(map_x[None, :], (out_h, out_w)).copy()
map_y = np.broadcast_to(map_y[:, None], (out_h, out_w)).copy()
output = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=config.border_rgb)
```

Derive valid pixels from finite/in-bounds coordinates and the optional nearest-neighbor-remapped mask. Record requested/effective angles, clipping, scale, output shape, and local maximum stretch.

- [ ] **Step 5: Add a synthetic cylindrical render/recovery test**

Generate a flat grid with text-like bars, render it into a cylinder source using the forward correspondence, unwrap it, and compare only valid interior pixels. Assert straight recovered vertical features and bounded mean absolute interpolation error; do not infer OCR quality.

- [ ] **Step 6: Integrate explicit/estimated geometry into the pipeline**

Prefer valid explicit geometry. Otherwise normalize the mask into full-image coordinates, estimate geometry over label rows, and unwrap from the full oriented image so cropping never removes source samples. On all expected reliability failures, retain A/B and add a stable reason.

- [ ] **Step 7: Run geometry and full suites**

Run: `cd modules/wine_label_preprocessing && python -m pytest tests/test_geometry.py tests/test_pipeline.py -v && python -m pytest -q`

Expected: PASS.

---

### Task 4: Optional lazy DewarpNet adapter

**Files:**
- Create: `modules/wine_label_preprocessing/src/wine_label_preprocessing/dewarpnet.py`
- Modify: `modules/wine_label_preprocessing/src/wine_label_preprocessing/models.py`
- Modify: `modules/wine_label_preprocessing/src/wine_label_preprocessing/pipeline.py`
- Modify: `modules/wine_label_preprocessing/src/wine_label_preprocessing/__init__.py`
- Test: `modules/wine_label_preprocessing/tests/test_dewarpnet.py`

**Interfaces:**
- Consumes: an RGB crop and `DewarpNetConfig(official_repo, wc_checkpoint, bm_checkpoint, device, align_corners)`.
- Produces: cached `DewarpNetAdapter`, `DewarpNetResult`, explicit unavailable/failure status, original-resolution remap, and pipeline variant E.

- [ ] **Step 1: Write failing tests for laziness, missing prerequisites, map scaling, and caching**

```python
def test_constructor_is_lazy(tmp_path: Path) -> None:
    adapter = DewarpNetAdapter(DewarpNetConfig(tmp_path / "repo", tmp_path / "wc.pkl", tmp_path / "bm.pkl"))
    assert not adapter.loaded

def test_missing_checkpoints_are_unavailable_not_resize(tmp_path: Path) -> None:
    config = DewarpNetConfig(tmp_path / "repo", tmp_path / "wc.pkl", tmp_path / "bm.pkl")
    result = DewarpNetAdapter(config).unwrap(np.zeros((40, 60, 3), np.uint8))
    assert result.image is None
    assert result.status == "unavailable"
    assert "checkpoint" in result.reason

def test_normalized_grid_maps_original_resolution() -> None:
    axis = np.linspace(-1.0, 1.0, 128, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(axis, axis)
    grid = np.stack((grid_x, grid_y), axis=-1)
    map_x, map_y, valid = normalized_grid_to_pixel_maps(grid, width=321, height=197, align_corners=False)
    assert map_x.shape == (197, 321)
    assert map_y.shape == (197, 321)
    assert valid.shape == (197, 321)
```

- [ ] **Step 2: Run focused tests and verify failures**

Run: `cd modules/wine_label_preprocessing && python -m pytest tests/test_dewarpnet.py -v`

Expected: FAIL because the adapter does not exist.

- [ ] **Step 3: Implement lazy prerequisite checks and official model loading**

Use `importlib.import_module` for torch and torchvision only inside `_ensure_loaded`. Load the official checkout's `models/__init__.py` under a package-unique module name with `spec_from_file_location(..., submodule_search_locations=[...])`, create `unetnc` and `dnetccnl`, remove an optional `module.` checkpoint prefix, call `eval`, and cache both models. Any exception is retained as a concise unavailable reason unless strict mode is requested.

- [ ] **Step 4: Implement reference preprocessing and original-resolution sampling**

```python
model_input = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)[..., ::-1]
tensor = torch.from_numpy(model_input.transpose(2, 0, 1).copy()).float().unsqueeze(0).div_(255.0)
world = torch.nn.functional.hardtanh(wc_model(tensor), 0.0, 1.0)
bm_input = torch.nn.functional.interpolate(world, size=(128, 128), mode="bilinear", align_corners=False)
normalized_map = bm_model(bm_input)[0].detach().cpu().numpy().transpose(1, 2, 0)
```

Resize the two coordinate channels to the original output width/height, optionally reproduce the official 3x3 map blur, convert normalized coordinates using the configured alignment convention, and call `cv2.remap` once on the original RGB crop. Return a bounds-derived valid mask and inference/load timings.

- [ ] **Step 5: Integrate the injected adapter into preprocess**

Only call it when supplied. Preserve unavailable/failed status and reason in metadata, never substitute another image, and expose its image/mask on success.

- [ ] **Step 6: Run DewarpNet and full core suites without PyTorch**

Run: `cd modules/wine_label_preprocessing && python -m pytest tests/test_dewarpnet.py -v && python -m pytest -q`

Expected: PASS without importing or downloading PyTorch during ordinary tests.

---

### Task 5: Comparison variants, injected evaluators, metrics, and timing

**Files:**
- Create: `modules/wine_label_preprocessing/src/wine_label_preprocessing/metrics.py`
- Create: `modules/wine_label_preprocessing/src/wine_label_preprocessing/benchmark.py`
- Modify: `modules/wine_label_preprocessing/src/wine_label_preprocessing/models.py`
- Modify: `modules/wine_label_preprocessing/src/wine_label_preprocessing/__init__.py`
- Test: `modules/wine_label_preprocessing/tests/test_metrics.py`
- Test: `modules/wine_label_preprocessing/tests/test_benchmark.py`

**Interfaces:**
- Consumes: `preprocess`, one optional OCR backend instance, one optional embedding backend instance, fixed catalog embeddings, and validated samples.
- Produces: A–E `ComparisonVariant` values, CER/exact/recall/degradation metrics, split leakage validation, and warm p50/p95 timing summaries.

```python
@dataclass(frozen=True, slots=True)
class BenchmarkSample:
    image_path: str | Path
    series_id: str
    split: Literal["tuning", "test"]
    annotations: ImageAnnotations | None = None
    conditions: tuple[str, ...] = ()
    truth: OCRTruth | None = None
    catalog_id: str | None = None

@dataclass(slots=True)
class ComparisonVariant:
    code: Literal["A", "B", "C", "D", "E"]
    image: np.ndarray | None
    status: Literal["ok", "skipped", "unavailable", "failed", "not_run"]
    reason: str | None = None
```

- [ ] **Step 1: Write failing metric and leakage tests**

```python
def test_cer_and_empty_truth() -> None:
    assert character_error_rate("wine", "wane") == pytest.approx(0.25)
    assert character_error_rate("", "") == 0.0
    assert character_error_rate("", "x") == 1.0

def test_recall_uses_fixed_cosine_gallery() -> None:
    gallery = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])}
    assert recall_at_k(np.array([0.9, 0.1]), "a", gallery, (1, 5)) == {1: True, 5: True}

def test_series_cannot_cross_tuning_and_test() -> None:
    samples = [
        BenchmarkSample("a.jpg", "series-1", "tuning"),
        BenchmarkSample("b.jpg", "series-1", "test"),
    ]
    with pytest.raises(ManifestError, match="series"):
        validate_samples(samples)
```

- [ ] **Step 2: Implement text and retrieval metrics**

Implement Unicode NFKC normalization, case folding, whitespace collapse, an iterative two-row Levenshtein distance, exact field flags, vectorized cosine ranking with zero-vector validation, per-condition grouping, and paired degradation relative to A.

- [ ] **Step 3: Write failing benchmark tests with counting fake backends**

```python
class CountingOCRBackend:
    name = "counting-ocr"

    def __init__(self) -> None:
        self.instance_id = id(self)
        self.calls = 0

    def recognize(self, image: np.ndarray) -> OCRPrediction:
        self.calls += 1
        return OCRPrediction(full_text="", name="", producer="", year="")

class CountingEmbeddingBackend:
    name = "counting-embedding"

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, image: np.ndarray) -> np.ndarray:
        self.calls += 1
        return np.array([1.0, 0.0], dtype=np.float32)

def test_one_backend_instance_is_reused_for_every_available_variant() -> None:
    ocr = CountingOCRBackend()
    embedder = CountingEmbeddingBackend()
    sample = BenchmarkSample("query.png", "series-1", "test")
    report = run_benchmark([sample], BenchmarkConfig(), ocr_backend=ocr, embedding_backend=embedder, gallery={})
    assert set(report["variants"]) == {"A", "B", "C", "D", "E"}
    assert ocr.instance_id == report["ocr_backend"]["instance_id"]
    assert report["variants"]["E"]["status"] == "not_run"
```

- [ ] **Step 4: Implement A–E creation and benchmark protocols**

Create A from the exact crop, B by applying `mild_ocr_profile`, C from cylindrical output, D by applying the same profile to C, and E only from successful DewarpNet output. Preserve missing reasons. Define protocols whose `name` identifies the single reused evaluator instance.

- [ ] **Step 5: Implement warm timing summaries**

Decode once, execute configurable warmup runs, measure in-memory preprocessing repetitions with `perf_counter_ns`, and compute percentiles with NumPy. Record sample count, warmup count, repeat count, included stages, excluded decode/model-load/visualization/write stages, and environment versions/hardware.

- [ ] **Step 6: Run metric, benchmark, and full suites**

Run: `cd modules/wine_label_preprocessing && python -m pytest tests/test_metrics.py tests/test_benchmark.py -v && python -m pytest -q`

Expected: PASS.

---

### Task 6: Visualization, JSON output, and file/folder CLI

**Files:**
- Create: `modules/wine_label_preprocessing/src/wine_label_preprocessing/visualization.py`
- Create: `modules/wine_label_preprocessing/src/wine_label_preprocessing/reporting.py`
- Create: `modules/wine_label_preprocessing/src/wine_label_preprocessing/cli.py`
- Create: `modules/wine_label_preprocessing/src/wine_label_preprocessing/__main__.py`
- Test: `modules/wine_label_preprocessing/tests/test_cli.py`
- Test: `modules/wine_label_preprocessing/tests/test_visualization.py`

**Interfaces:**
- Consumes: comparison outputs and JSONL annotations/benchmark samples.
- Produces: aspect-preserving A–E contact sheets, per-image assets/JSON, aggregate `report.json`, `process`, and `benchmark` commands.

`make_contact_sheet` accepts `Mapping[str, ComparisonVariant]` and returns RGB `uint8`. `write_comparison` accepts an output directory, input-relative stem, variants, masks, and metadata and returns only JSON-compatible asset paths.

- [ ] **Step 1: Write failing visualization and CLI tests**

```python
def test_contact_sheet_marks_missing_variant() -> None:
    variants = {
        "A": ComparisonVariant("A", np.zeros((20, 30, 3), np.uint8), "ok", None),
        "E": ComparisonVariant("E", None, "unavailable", "DewarpNet not configured"),
    }
    sheet = make_contact_sheet(variants)
    assert sheet.ndim == 3 and sheet.shape[2] == 3
    assert sheet.dtype == np.uint8

def test_process_file_writes_variants_and_report(tmp_path: Path) -> None:
    input_path = tmp_path / "input.png"
    Image.fromarray(np.full((20, 30, 3), 127, np.uint8), mode="RGB").save(input_path)
    output = tmp_path / "out"
    assert main(["process", str(input_path), "--output-dir", str(output)]) == 0
    assert (output / "input" / "A_original.png").is_file()
    report = json.loads((output / "report.json").read_text())
    assert report["schema_version"] == "1.0"
```

- [ ] **Step 2: Implement OpenCV-only contact sheets**

Choose a fixed tile canvas, scale each RGB image down without distortion, center it, add a header band with `cv2.putText`, and render unavailable reason text on a neutral tile. Convert RGB/BGR only at file-write boundaries.

- [ ] **Step 3: Implement deterministic JSON serialization and assets**

Convert dataclasses, enums, paths, NumPy scalars, and arrays used as scalar metadata to JSON-compatible values. Never embed image arrays. Save variants losslessly as PNG, boolean masks as 0/255 PNG, and contact sheets. Preserve relative input paths and reject output collisions.

- [ ] **Step 4: Implement process CLI for a file or directory**

Parse single-file bbox/geometry flags and optional JSONL manifests. Folder discovery is limited to documented image suffixes and sorted for deterministic output. Without annotations, run baseline variants and report geometry unavailable. A per-image expected skip must not fail the run.

- [ ] **Step 5: Implement benchmark CLI**

Validate the JSONL manifest, generate variants, warm and time preprocessing, write metrics as `not_run` because the packaged CLI has no model backends, and include exact reasons. Reject series leakage with exit code 2 and an actionable message.

- [ ] **Step 6: Run CLI, visualization, and full suites**

Run: `cd modules/wine_label_preprocessing && python -m pytest tests/test_cli.py tests/test_visualization.py -v && python -m pytest -q`

Expected: PASS.

---

### Task 7: README, synthetic evidence, and final verification

**Files:**
- Create: `modules/wine_label_preprocessing/README.md`
- Modify: `modules/wine_label_preprocessing/src/wine_label_preprocessing/__init__.py`
- Modify: tests from Tasks 1–6 only if final verification exposes a real defect

**Interfaces:**
- Consumes: all completed package APIs and CLI commands.
- Produces: installation/integration documentation, coordinate/color guarantees, synthetic validation command, DewarpNet blocker details, and honest final test/timing evidence.

- [ ] **Step 1: Write the README with runnable commands**

Document editable installation, `process` file/folder commands, bbox/geometry examples, JSONL schemas, Python API, RGB contract, safe defaults, A–E meaning, benchmark backend injection, output layout, and exact limitations. Include official DewarpNet repository/paper links, required world-coordinate/backward-map checkpoints, `256x256`/`128x128` map handling, optional install command, and domain-shift warning.

- [ ] **Step 2: Run package build and complete tests**

Run: `cd modules/wine_label_preprocessing && python -m pytest -q && python -m build`

Expected: all tests pass and sdist/wheel build. If `build` is not installed, use `python -m pip wheel . --no-deps -w /tmp/wine-label-preprocessing-wheel` and record that substitution.

- [ ] **Step 3: Run a synthetic CLI comparison and benchmark**

Generate the synthetic cylindrical grid used by tests into a temporary directory, invoke `wine-label-preprocess process` with explicit geometry, and inspect that A–D images, valid mask, contact sheet, and JSON report exist. Run the benchmark with at least one warmup and five measured repetitions.

- [ ] **Step 4: Record measured evidence without overclaiming**

Report the actual machine/CPU description, image dimensions, included timing stages, warmup/repetition counts, p50/p95, test count, and DewarpNet availability. State explicitly that OCR CER/exact fields and Recall@1/5 were not measured because the repository has no real dataset/backends/gallery.

- [ ] **Step 5: Final hygiene check**

Run: `git diff --check && git status --short`

Expected: only intended source/tests/README and the uncommitted design/plan documentation are present; no generated build artifacts, caches, downloaded weights, or official DewarpNet checkout are added to Git.
