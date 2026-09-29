import csv

import numpy as np
import pytest

from dinov2_retrieval.wine_catalog.publish_references import MODEL, publish


class Cursor:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return self.rows


class Connection:
    def __init__(self, rows):
        self.rows = rows
        self.writes = []
        self.committed = False

    def execute(self, query, params=None):
        if params is None:
            return Cursor(self.rows)
        self.writes.append(params)

    def commit(self):
        self.committed = True


def setup_data(root):
    (root / "exports/dataset_photos").mkdir(parents=True)
    (root / "exports/dataset_photos/photo.jpg").write_bytes(b"image")
    (root / "web_photos").mkdir()
    with (root / "exports/dataset_photo_manifest.csv").open("w") as target:
        writer = csv.writer(target)
        writer.writerow(["dataset_photo", "relative_path", "status"])
        writer.writerow(["original.jpg", "photo.jpg", "matched"])


def row(identity, slug, canonical, dimension=1536):
    return (identity, slug, canonical, "original.jpg", np.ones(dimension), MODEL,
            None, None, None)


def test_publish_canonical_ids_preserves_shared_photos_and_is_repeatable(tmp_path):
    setup_data(tmp_path)
    c = Connection([row(1, "alpha", 1), row(2, "alpha", 1), row(3, "beta", 3)])
    result = publish(c, tmp_path)
    assert result == {"reference_images": 2, "distinct_slugs": 2}
    assert {params[0] for params in c.writes} == {"1", "3"}
    assert len({params[2] for params in c.writes}) == 2
    assert c.committed
    assert publish(c, tmp_path) == result


def test_wrong_dimension_has_no_database_writes(tmp_path):
    setup_data(tmp_path)
    c = Connection([row(1, "alpha", 1, dimension=384)])
    with pytest.raises(ValueError, match="dimension"):
        publish(c, tmp_path)
    assert c.writes == []


def test_empty_publication_rejected(tmp_path):
    setup_data(tmp_path)
    with pytest.raises(ValueError, match="empty"):
        publish(Connection([]), tmp_path)
