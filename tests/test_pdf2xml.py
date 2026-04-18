"""Tests for ecgtizer/PDF2XML.py

Tests signal processing functions:
- sup_holes: fill missing points in extracted signals
- lead_extraction: digitize signals from track images
- clean_tracks: remove noise from track images
- check_noise_type: detect image type and noise level
"""
import numpy as np
import cv2
import pytest

from ecgtizer.PDF2XML import sup_holes


class TestSupHoles:

    def test_signal_without_holes(self):
        """Signal with no zeros should be returned unchanged (minus last element)."""
        signal = [10, 20, 30, 40, 50]
        result = sup_holes(signal, 'classic')
        np.testing.assert_array_equal(result, [10, 20, 30, 40])

    def test_hole_at_beginning(self):
        """Hole at start should be filled with next non-zero value."""
        signal = [0, 0, 30, 40, 50]
        result = sup_holes(signal, 'classic')
        assert result[0] == 30
        assert result[1] == 30

    def test_hole_at_end(self):
        """Hole at end should be filled with previous non-zero value."""
        signal = [10, 20, 30, 0, 0]
        result = sup_holes(signal, 'classic')
        assert result[-1] != 0

    def test_hole_in_middle(self):
        """Interior hole should be filled with mean of neighbors."""
        signal = [10, 0, 30, 40, 50]
        result = sup_holes(signal, 'classic')
        # Middle hole between 10 and 30 should be mean = 20
        assert result[1] == pytest.approx(20)

    def test_constant_signal_returns_zeros(self):
        """All-same signal (diff == 0) should return zeros."""
        signal = np.array([5, 5, 5, 5, 5])
        result = sup_holes(signal, 'classic')
        np.testing.assert_array_equal(result, np.zeros(5))

    def test_all_zeros_returns_zeros(self):
        """All-zero signal (constant) should be replaced with zeros."""
        signal = np.array([0, 0, 0, 0, 0])
        result = sup_holes(signal, 'classic')
        np.testing.assert_array_equal(result, np.zeros(5))

    def test_numpy_array_input(self):
        """Should work with numpy array input."""
        signal = np.array([10.0, 0.0, 30.0, 0.0, 50.0])
        result = sup_holes(signal, 'classic')
        assert len(result) == 4
        assert result[1] != 0

    def test_single_nonzero_value(self):
        """Signal with only one non-zero value."""
        signal = np.array([0, 0, 42, 0, 0])
        result = sup_holes(signal, 'classic')
        # First and last filled, middle holes interpolated
        assert result[0] == 42
        assert all(v != 0 for v in result)

    def test_large_signal(self):
        """Performance check with a large signal."""
        rng = np.random.RandomState(42)
        signal = rng.randint(1, 1000, size=10000).astype(float)
        # Insert some holes
        signal[100:110] = 0
        signal[5000:5005] = 0
        result = sup_holes(signal, 'classic')
        assert len(result) == 9999
        # Holes should be filled
        assert all(result[100:110] != 0)

    def test_nan_values_are_filled(self):
        """NaN values should be treated as holes and interpolated."""
        signal = np.array([10.0, np.nan, np.nan, 40.0, 50.0])
        result = sup_holes(signal, 'classic')
        assert not np.any(np.isnan(result))
        # Interpolated values should be between neighbours
        assert 10.0 < result[1] < 40.0
        assert 10.0 < result[2] < 40.0

    def test_nan_at_boundaries(self):
        """NaN at start and end should be filled from nearest valid values."""
        signal = np.array([np.nan, np.nan, 30.0, 40.0, np.nan])
        result = sup_holes(signal, 'classic')
        assert not np.any(np.isnan(result))

    def test_mixed_nan_and_zeros(self):
        """Both NaN and zero holes should be filled."""
        signal = np.array([10.0, 0.0, np.nan, 40.0, 0.0, 60.0])
        result = sup_holes(signal, 'classic')
        assert not np.any(np.isnan(result))
        assert all(v != 0 for v in result)


class TestLeadExtraction:

    def test_lazy_extraction_on_tracks(self, sample_binary_image):
        """lead_extraction with lazy method should produce output for each track."""
        from ecgtizer.PDF2XML import lead_extraction
        dic_tracks = {0: sample_binary_image}
        result, image_bins, not_scaled = lead_extraction(
            dic_tracks, "lazy", "wellue", False, False
        )
        assert 0 in result
        assert len(result[0]) > 0

    def test_full_extraction_on_tracks(self, sample_binary_image):
        from ecgtizer.PDF2XML import lead_extraction
        dic_tracks = {0: sample_binary_image}
        result, image_bins, not_scaled = lead_extraction(
            dic_tracks, "full", "wellue", False, False
        )
        assert 0 in result

    def test_fragmented_extraction_on_tracks(self, sample_binary_image):
        from ecgtizer.PDF2XML import lead_extraction
        dic_tracks = {0: sample_binary_image}
        result, image_bins, not_scaled = lead_extraction(
            dic_tracks, "fragmented", "wellue", False, False
        )
        assert 0 in result

    def test_extraction_output_is_scaled(self, sample_binary_image):
        """Scaled output should have ~5000 points for non-kardia/non-classic."""
        from ecgtizer.PDF2XML import lead_extraction
        dic_tracks = {0: sample_binary_image}
        result, _, not_scaled = lead_extraction(
            dic_tracks, "full", "wellue", False, False
        )
        assert len(result[0]) == 5000 or len(result[0]) > 4000

    def test_multiple_tracks(self, sample_binary_image):
        """Should handle multiple track images."""
        from ecgtizer.PDF2XML import lead_extraction
        dic_tracks = {0: sample_binary_image, 1: sample_binary_image.copy()}
        result, _, _ = lead_extraction(dic_tracks, "full", "wellue", False, False)
        assert len(result) == 2


class TestCheckNoiseType:

    def test_returns_type_and_noise(self, sample_color_image):
        from ecgtizer.PDF2XML import check_noise_type
        typ, noise = check_noise_type(sample_color_image, 300, False)
        assert isinstance(typ, str)
        assert isinstance(noise, (bool, float))

    def test_white_background_classic(self):
        """A white image with varied color content should be detected as classic."""
        from ecgtizer.PDF2XML import check_noise_type
        h, w = 600, 800
        img = np.ones((h, w, 3), dtype=np.uint8) * 255
        # Add varied-color content at the middle column so liste has >1 entries
        # (otherwise the function detects single-color as Kardia)
        for y in range(0, h, 20):
            color_val = (y * 3) % 256
            img[y, w // 2] = [255, color_val, color_val]
        # Add dark signal
        img[200:210, 100:700] = [0, 0, 0]
        typ, noise = check_noise_type(img, 300, False)
        assert typ.lower() in ('classic', 'wellue', 'apple')

    def test_black_background_kardia(self):
        """A mostly dark image should be detected as Kardia type."""
        from ecgtizer.PDF2XML import check_noise_type
        h, w = 600, 800
        img = np.zeros((h, w, 3), dtype=np.uint8)
        # Add white ECG line
        img[300, :] = [255, 255, 255]
        typ, noise = check_noise_type(img, 300, False)
        assert typ.lower() == 'kardia'
