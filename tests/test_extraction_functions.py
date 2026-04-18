"""Tests for ecgtizer/extraction_functions.py

Tests all three signal extraction algorithms:
- lazy_extraction: follows closest lit pixel
- full_extraction: averages lit pixel positions per column
- fragmented_extraction: handles text/noise by grouping lit pixels
"""
import numpy as np
import pytest
from ecgtizer.extraction_functions import lazy_extraction, full_extraction, fragmented_extraction


class TestLazyExtraction:

    def test_returns_list(self, sample_binary_image):
        result = lazy_extraction(sample_binary_image)
        assert isinstance(result, list)

    def test_output_length_matches_image_width(self, sample_binary_image):
        result = lazy_extraction(sample_binary_image)
        assert len(result) == sample_binary_image.shape[1]

    def test_follows_sine_wave_shape(self, sample_binary_image):
        """Signal should roughly follow the sine wave drawn in the fixture."""
        result = lazy_extraction(sample_binary_image)
        signal = np.array(result)
        h = sample_binary_image.shape[0]
        # Signal should oscillate around the center
        assert abs(np.mean(signal) - h / 2) < 20

    def test_all_black_image_uses_midpoint(self):
        """If no lit pixels in first column, anchor defaults to midpoint."""
        img = np.zeros((100, 50), dtype=np.uint8)
        result = lazy_extraction(img)
        assert len(result) == 50
        # All values should be the midpoint since no pixels are lit
        assert result[0] == 50

    def test_single_horizontal_line(self):
        """A single horizontal line should produce a constant signal."""
        h, w = 100, 200
        img = np.zeros((h, w), dtype=np.uint8)
        y_pos = 40
        img[y_pos, :] = 255
        result = lazy_extraction(img)
        assert all(v == y_pos for v in result)

    def test_handles_noisy_image(self, sample_binary_image_with_noise):
        """Should still produce a signal even with noise pixels."""
        result = lazy_extraction(sample_binary_image_with_noise)
        assert len(result) == sample_binary_image_with_noise.shape[1]


class TestFullExtraction:

    def test_returns_numpy_array(self, sample_binary_image):
        result = full_extraction(sample_binary_image)
        assert isinstance(result, np.ndarray)

    def test_output_length_matches_image_width(self, sample_binary_image):
        result = full_extraction(sample_binary_image)
        assert len(result) == sample_binary_image.shape[1]

    def test_follows_sine_wave_shape(self, sample_binary_image):
        result = full_extraction(sample_binary_image)
        h = sample_binary_image.shape[0]
        # Mean should be around center
        assert abs(np.mean(result) - h / 2) < 20

    def test_empty_columns_return_near_zero(self):
        """Columns with no lit pixels should return near-zero values."""
        img = np.zeros((100, 10), dtype=np.uint8)
        result = full_extraction(img)
        assert all(abs(v) < 1 for v in result)

    def test_single_pixel_per_column(self):
        """If exactly one pixel is lit per column, result should be its position."""
        h, w = 100, 50
        img = np.zeros((h, w), dtype=np.uint8)
        positions = [30, 40, 50, 60, 70]
        for x in range(w):
            y = positions[x % len(positions)]
            img[y, x] = 255
        result = full_extraction(img)
        for x in range(w):
            expected = positions[x % len(positions)]
            assert abs(result[x] - expected) < 2

    def test_multiple_lit_pixels_averages(self):
        """Multiple lit pixels in a column should be averaged."""
        h, w = 100, 5
        img = np.zeros((h, w), dtype=np.uint8)
        # Light up pixels at positions 20 and 40 in column 0
        img[20, 0] = 255
        img[40, 0] = 255
        result = full_extraction(img)
        # Average of 20 and 40 = 30
        assert abs(result[0] - 30) < 1


class TestFragmentedExtraction:

    def test_returns_list(self, sample_binary_image):
        result = fragmented_extraction(sample_binary_image)
        assert isinstance(result, list)

    def test_output_length_matches_image_width(self, sample_binary_image):
        result = fragmented_extraction(sample_binary_image)
        assert len(result) == sample_binary_image.shape[1]

    def test_handles_text_noise_at_top(self, sample_binary_image_with_noise):
        """Should extract signal from the last pixel group, ignoring top noise."""
        result = fragmented_extraction(sample_binary_image_with_noise)
        signal = np.array(result)
        h = sample_binary_image_with_noise.shape[0]
        # The signal center should be around h/2, not up at the noise region (10-30)
        mean_val = np.mean(signal)
        assert mean_val > 50  # Should be in the lower half, not at the top noise

    def test_all_black_image(self):
        """Image with no lit pixels should still return a signal."""
        img = np.zeros((100, 20), dtype=np.uint8)
        result = fragmented_extraction(img)
        assert len(result) == 20

    def test_no_nan_on_empty_columns(self):
        """Columns with no lit pixels must not produce NaN values."""
        h, w = 100, 50
        img = np.zeros((h, w), dtype=np.uint8)
        # Draw signal only on even columns, leave odd columns empty
        for x in range(0, w, 2):
            img[50, x] = 255
        result = fragmented_extraction(img)
        assert len(result) == w
        assert not any(np.isnan(v) for v in result)

    def test_no_nan_on_sparse_image(self):
        """Realistic sparse image with many empty columns produces no NaN."""
        h, w = 200, 500
        img = np.zeros((h, w), dtype=np.uint8)
        # Draw a thin sine wave with gaps
        for x in range(w):
            if x % 3 == 0:  # skip every 3rd column
                continue
            y = int(h / 2 + 30 * np.sin(2 * np.pi * x / 200))
            img[y, x] = 255
        result = fragmented_extraction(img)
        assert len(result) == w
        assert not any(np.isnan(v) for v in result)

    def test_single_group_per_column(self):
        """With a single contiguous group, should average it correctly."""
        h, w = 100, 10
        img = np.zeros((h, w), dtype=np.uint8)
        # Contiguous block from 45-55
        for x in range(w):
            for y in range(45, 56):
                img[y, x] = 255
        result = fragmented_extraction(img)
        for val in result:
            assert abs(val - 50) < 2


class TestExtractionConsistency:
    """Cross-method comparison tests."""

    def test_all_methods_same_length(self, sample_binary_image):
        lazy = lazy_extraction(sample_binary_image)
        full = full_extraction(sample_binary_image)
        frag = fragmented_extraction(sample_binary_image)
        assert len(lazy) == len(full) == len(frag)

    def test_all_methods_similar_mean(self, sample_binary_image):
        """All methods should produce similar mean values for a clean signal."""
        lazy = np.array(lazy_extraction(sample_binary_image))
        full = np.array(full_extraction(sample_binary_image))
        frag = np.array(fragmented_extraction(sample_binary_image))
        means = [np.mean(lazy), np.mean(full), np.mean(frag)]
        assert max(means) - min(means) < 30

    def test_clean_image_high_correlation(self, sample_binary_image):
        """On a clean image, all three methods should correlate highly."""
        lazy = np.array(lazy_extraction(sample_binary_image), dtype=float)
        full = np.array(full_extraction(sample_binary_image), dtype=float)
        corr = np.corrcoef(lazy, full)[0, 1]
        assert corr > 0.8
