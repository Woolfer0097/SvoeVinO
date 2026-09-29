"""Resumable DINOv2 image and multilingual E5 text embeddings for wines."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from ..config import DEFAULT_DINO_MODEL_NAME
from .migrations import apply_migrations
from .photo_manifest import read_photo_manifest

DEFAULT_IMAGE_MODEL = DEFAULT_DINO_MODEL_NAME
DEFAULT_TEXT_MODEL = "intfloat/multilingual-e5-base"
KINDS = ("dataset_photo", "web_photo", "description_text")
FIELDS = (
    ("Название", "name"),
    ("Категория", "category"),
    ("Цвет", "color"),
    ("Регион", "region"),
    ("Сорт винограда", "grape_variety"),
    ("Описание", "description"),
    ("Винодельня", "winery"),
)


def wine_text(values: dict[str, str | None]) -> str:
    """Build the passage from the seven CSV wine attributes, in stable order."""

    return "\n".join(
        f"{label}: {value.strip()}"
        for label, field in FIELDS
        if (value := values.get(field)) and value.strip()
    )


class E5TextEmbedder:
    """Mean-pool and normalize passages exactly as in the E5 model card."""

    def __init__(self, model_name: str = DEFAULT_TEXT_MODEL, device: str = "auto") -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.model_name = model_name
        self.device = _device(torch, device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        tokens = self.tokenizer(
            [f"passage: {text}" for text in texts],
            max_length=512, padding=True, truncation=True, return_tensors="pt",
        ).to(self.device)
        with self.torch.inference_mode():
            hidden = self.model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).bool()
            pooled = hidden.masked_fill(~mask, 0.0).sum(dim=1) / mask.sum(dim=1)
            normalized = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
        return normalized.float().cpu().tolist()


def _device(torch, requested: str):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def generate_embeddings(
    connection,
    *,
    kinds: tuple[str, ...] = KINDS,
    uploads_root: Path | None = None,
    manifest_path: Path | None = None,
    web_photos_root: Path | None = None,
    image_model: str = DEFAULT_IMAGE_MODEL,
    text_model: str = DEFAULT_TEXT_MODEL,
    device: str = "auto",
    image_dtype: str = "float32",
    text_batch_size: int = 16,
    limit: int | None = None,
    retry_failed: bool = False,
) -> dict[str, dict[str, int]]:
    """Fill only missing/model-mismatched vectors; every row commits separately."""

    if not set(kinds).issubset(KINDS) or text_batch_size <= 0:
        raise ValueError("Invalid embedding kinds or text batch size")
    if image_dtype not in {"float32", "float16"}:
        raise ValueError("image_dtype must be float32 or float16")
    apply_migrations(connection)
    from pgvector.psycopg import register_vector

    register_vector(connection)
    results: dict[str, dict[str, int]] = {}
    for kind in kinds:
        model_name = text_model if kind == "description_text" else image_model
        rows = _pending_rows(connection, kind, model_name, limit, retry_failed)
        result = Counter({"embedded": 0, "errors": 0, "selected": len(rows)})
        if not rows:
            results[kind] = dict(result)
            continue
        if kind == "dataset_photo":
            if uploads_root is None or manifest_path is None:
                raise ValueError("dataset_photo requires --uploads-root and --photo-manifest")
            root = uploads_root.expanduser().resolve(strict=True)
            manifest = read_photo_manifest(manifest_path)
        elif kind == "web_photo":
            if web_photos_root is None:
                raise ValueError("web_photo requires --web-photos-root")
            root = web_photos_root.expanduser().resolve(strict=True)
            manifest = {}
        else:
            root = None
            manifest = {}

        if kind == "description_text":
            embedder = E5TextEmbedder(model_name, device)
            for offset in range(0, len(rows), text_batch_size):
                batch = rows[offset:offset + text_batch_size]
                texts = [wine_text(row) for row in batch]
                usable = [(row, text) for row, text in zip(batch, texts) if text]
                for row, text in zip(batch, texts):
                    _started(connection, row["id"], kind, model_name)
                    if not text:
                        _failure(connection, row["id"], kind, model_name, "No CSV text to embed")
                        result["errors"] += 1
                if not usable:
                    continue
                try:
                    vectors = embedder.embed_many([text for _, text in usable])
                    if len(vectors) != len(usable):
                        raise RuntimeError("Text model returned an unexpected number of vectors")
                except Exception as exc:
                    for row, _ in usable:
                        _failure(connection, row["id"], kind, model_name, str(exc))
                        result["errors"] += 1
                    continue
                for (row, _), vector in zip(usable, vectors):
                    _success(connection, row["id"], kind, model_name, vector)
                    result["embedded"] += 1
        else:
            from ..embedding.dinov2_embedder import DinoV2Embedder
            from ..preprocessing.image_preprocessor import ImagePreprocessor
            from transformers import AutoConfig

            if device == "cuda":
                import torch
                _device(torch, device)
            dimension = AutoConfig.from_pretrained(model_name).hidden_size
            embedder = DinoV2Embedder(
                model_name, embedding_dimension=dimension,
                device=None if device == "auto" else device,
                dtype=image_dtype,
            )
            if device != "auto" and embedder.device != device:
                raise RuntimeError(f"Image embedder selected {embedder.device}, requested {device}")
            cache: dict[Path, list[float]] = {}
            preprocessor = ImagePreprocessor()
            for row in rows:
                _started(connection, row["id"], kind, model_name)
                try:
                    filename = row[kind]
                    if not filename:
                        raise FileNotFoundError(f"No {kind} on wine row")
                    if kind == "dataset_photo":
                        relative = manifest.get(filename)
                        if relative is None:
                            raise FileNotFoundError(f"No unambiguous manifest entry for {filename}")
                    else:
                        relative = Path(filename).name
                    path = (root / relative).resolve()
                    if not path.is_relative_to(root) or not path.is_file():
                        raise FileNotFoundError(f"Image not found under {root}: {relative}")
                    if path not in cache:
                        with preprocessor.open_rgb(path) as image:
                            cache[path] = embedder.embed(image)
                    _success(connection, row["id"], kind, model_name, cache[path])
                    result["embedded"] += 1
                except Exception as exc:
                    _failure(connection, row["id"], kind, model_name, str(exc))
                    result["errors"] += 1
        results[kind] = dict(result)
        # Do not keep one model alive while loading the next embedding kind.
        del embedder
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def _pending_rows(connection, kind: str, model_name: str, limit: int | None, retry_failed: bool):
    # Names are selected only from KINDS above; never interpolate user-supplied SQL.
    vector = f"{kind}_embedding"
    model = f"{kind}_embedding_model"
    fields = ", ".join(field for _, field in FIELDS)
    sql = (
        f"SELECT w.id, w.dataset_photo, w.web_photo, {fields} FROM wines w "
        f"WHERE (w.{vector} IS NULL OR w.{model} IS DISTINCT FROM %s)"
    )
    parameters: list[object] = [model_name]
    if kind in ("dataset_photo", "web_photo"):
        sql += f" AND w.{kind} IS NOT NULL"
    if retry_failed:
        sql += " AND EXISTS (SELECT 1 FROM wine_embedding_jobs j WHERE j.wine_id=w.id AND j.kind=%s AND j.status='failed')"
        parameters.append(kind)
    sql += " ORDER BY w.id"
    if limit is not None:
        sql += " LIMIT %s"
        parameters.append(limit)
    from psycopg.rows import dict_row

    with connection.cursor(row_factory=dict_row) as cursor:
        cursor.execute(sql, parameters)
        return cursor.fetchall()


def _started(connection, wine_id: int, kind: str, model_name: str) -> None:
    connection.execute(
        "INSERT INTO wine_embedding_jobs (wine_id, kind, status, model_name, attempts, updated_at) "
        "VALUES (%s, %s, 'in_progress', %s, 1, NOW()) "
        "ON CONFLICT (wine_id, kind) DO UPDATE SET status='in_progress', "
        "model_name=EXCLUDED.model_name, attempts=wine_embedding_jobs.attempts+1, "
        "last_error=NULL, updated_at=NOW()",
        (wine_id, kind, model_name),
    )


def _success(connection, wine_id: int, kind: str, model_name: str, values: list[float]) -> None:
    import numpy as np

    vector = f"{kind}_embedding"
    model = f"{kind}_embedding_model"
    with connection.transaction():
        connection.execute(
            f"UPDATE wines SET {vector}=%s, {model}=%s WHERE id=%s",
            (np.asarray(values, dtype=np.float32), model_name, wine_id),
        )
        connection.execute(
            "INSERT INTO wine_embedding_jobs (wine_id, kind, status, model_name, attempts, updated_at) "
            "VALUES (%s, %s, 'succeeded', %s, 1, NOW()) "
            "ON CONFLICT (wine_id, kind) DO UPDATE SET status='succeeded', "
            "model_name=EXCLUDED.model_name, "
            "last_error=NULL, updated_at=NOW()",
            (wine_id, kind, model_name),
        )


def _failure(connection, wine_id: int, kind: str, model_name: str, error: str) -> None:
    connection.execute(
        "INSERT INTO wine_embedding_jobs (wine_id, kind, status, model_name, attempts, last_error, updated_at) "
        "VALUES (%s, %s, 'failed', %s, 1, %s, NOW()) "
        "ON CONFLICT (wine_id, kind) DO UPDATE SET status='failed', "
        "model_name=EXCLUDED.model_name, "
        "last_error=EXCLUDED.last_error, updated_at=NOW()",
        (wine_id, kind, model_name, error[:4000]),
    )
