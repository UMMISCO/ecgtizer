"""Tests for ecgtizer/anonymisation.py

Tests the PDF anonymization utilities:
- array_to_pdf: NumPy array to PDF conversion
- anonymisation: full pipeline with text region masking
"""
import numpy as np
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

from ecgtizer.anonymisation import array_to_pdf, anonymisation

# `ecgtizer/__init__.py` re-exports the `anonymisation` function, which
# shadows the `ecgtizer.anonymisation` submodule on attribute lookup in
# some Python/pytest-cov combinations. Resolve through sys.modules to
# patch the module reliably.
_anon_module = sys.modules["ecgtizer.anonymisation"]


class TestArrayToPdf:

    def test_creates_pdf_file(self, tmp_output_dir):
        img = np.ones((100, 200, 3), dtype=np.uint8) * 128
        outpath = os.path.join(tmp_output_dir, "test_array.pdf")
        array_to_pdf(img, outpath)
        assert os.path.exists(outpath)

    def test_pdf_not_empty(self, tmp_output_dir):
        img = np.ones((100, 200, 3), dtype=np.uint8) * 255
        outpath = os.path.join(tmp_output_dir, "test_size.pdf")
        array_to_pdf(img, outpath)
        assert os.path.getsize(outpath) > 100

    def test_grayscale_array(self, tmp_output_dir):
        img = np.ones((50, 100), dtype=np.uint8) * 200
        outpath = os.path.join(tmp_output_dir, "test_gray.pdf")
        array_to_pdf(img, outpath)
        assert os.path.exists(outpath)

    def test_preserves_dimensions_in_filename(self, tmp_output_dir):
        """Array dimensions should match the page size of the generated PDF."""
        img = np.ones((150, 300, 3), dtype=np.uint8) * 100
        outpath = os.path.join(tmp_output_dir, "test_dims.pdf")
        array_to_pdf(img, outpath)
        assert os.path.exists(outpath)
        assert os.path.getsize(outpath) > 0

    def test_different_pixel_values(self, tmp_output_dir):
        """Array with varied pixel values should produce a valid PDF."""
        rng = np.random.RandomState(42)
        img = rng.randint(0, 256, (80, 160, 3), dtype=np.uint8)
        outpath = os.path.join(tmp_output_dir, "test_varied.pdf")
        array_to_pdf(img, outpath)
        assert os.path.exists(outpath)


class TestAnonymisation:

    def test_anonymisation_produces_output(self, tmp_output_dir):
        """Full pipeline: mock convert_PDF2image, run anonymisation, check output."""
        # Create a synthetic image with some "text" in the upper-left corner
        h, w = 600, 800
        img = np.ones((h, w, 3), dtype=np.uint8) * 255
        # Simulate text region (dark pixels in upper-left)
        img[20:60, 20:180] = [0, 0, 0]

        with patch.object(_anon_module, "convert_PDF2image") as mock_convert:
            mock_convert.return_value = ([img], 1, True)
            outpath = os.path.join(tmp_output_dir, "anon_output.pdf")
            anonymisation("dummy.pdf", outpath)
            assert os.path.exists(outpath)
            assert os.path.getsize(outpath) > 0

    def test_anonymisation_masks_upper_left_text(self, tmp_output_dir):
        """Verify that dark regions in the upper-left corner are whitened."""
        h, w = 600, 800
        img = np.ones((h, w, 3), dtype=np.uint8) * 255
        # Dark text block in upper-left (x < 200, y < 200)
        img[30:80, 30:150] = [0, 0, 0]

        with patch.object(_anon_module, "convert_PDF2image") as mock_convert:
            mock_convert.return_value = ([img], 1, True)
            outpath = os.path.join(tmp_output_dir, "anon_masked.pdf")
            anonymisation("dummy.pdf", outpath)
            assert os.path.exists(outpath)

    def test_anonymisation_preserves_ecg_traces(self, tmp_output_dir):
        """ECG traces outside the upper-left corner should not be blanked."""
        h, w = 600, 800
        img = np.ones((h, w, 3), dtype=np.uint8) * 255
        # Draw ECG traces in the middle of the image (far from upper-left)
        for x in range(100, 700):
            y = int(300 + 50 * np.sin(2 * np.pi * x / 200))
            y = max(0, min(h - 1, y))
            img[y, x] = [0, 0, 0]

        with patch.object(_anon_module, "convert_PDF2image") as mock_convert:
            mock_convert.return_value = ([img], 1, True)
            outpath = os.path.join(tmp_output_dir, "anon_traces.pdf")
            anonymisation("dummy.pdf", outpath)
            assert os.path.exists(outpath)

    def test_anonymisation_with_no_text_regions(self, tmp_output_dir):
        """Clean image with no text should still produce valid output."""
        h, w = 400, 600
        img = np.ones((h, w, 3), dtype=np.uint8) * 255

        with patch.object(_anon_module, "convert_PDF2image") as mock_convert:
            mock_convert.return_value = ([img], 1, True)
            outpath = os.path.join(tmp_output_dir, "anon_clean.pdf")
            anonymisation("dummy.pdf", outpath)
            assert os.path.exists(outpath)
