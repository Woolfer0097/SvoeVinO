from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from wine_ocr.reporting.filenames import safe_report_filename


class ReportingFilenameTests(unittest.TestCase):
    def test_safe_report_filename_sanitizes_uploaded_name(self) -> None:
        self.assertEqual(
            safe_report_filename("Grand Reserve 2023!.jpg", suffix="fixed"),
            "Grand-Reserve-2023-fixed.txt",
        )

    def test_safe_report_filename_falls_back_to_upload(self) -> None:
        self.assertEqual(
            safe_report_filename("!!!.jpg", suffix="fixed"),
            "upload-fixed.txt",
        )

    def test_safe_report_filename_generates_hex_suffix(self) -> None:
        filename = safe_report_filename(None)

        self.assertIsNotNone(re.fullmatch(r"upload-[0-9a-f]{32}\.txt", filename))


if __name__ == "__main__":
    unittest.main()
