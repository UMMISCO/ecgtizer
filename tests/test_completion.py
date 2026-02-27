"""Tests for ecgtizer/completion.py

Tests utility functions (unit-testable):
- linear_interpolation: resamples to 5000 points
- denormalization: reverse normalization
- normalization2: simple [-1, 1] normalization
- replace_random: prepare signal matrix for model input

Integration tests:
- Autoencoder_net architecture
- load_model + completion_ pipeline
"""
import numpy as np
import pytest


class TestLinearInterpolation:

    def test_output_length_is_5000(self):
        from ecgtizer.completion import linear_interpolation
        signal = np.sin(np.linspace(0, 2 * np.pi, 500))
        result = linear_interpolation(signal)
        assert len(result) == 5000

    def test_preserves_constant_signal(self):
        from ecgtizer.completion import linear_interpolation
        signal = np.ones(100) * 42
        result = linear_interpolation(signal)
        np.testing.assert_allclose(result, 42, atol=1e-10)

    def test_preserves_range(self):
        from ecgtizer.completion import linear_interpolation
        signal = np.sin(np.linspace(0, 4 * np.pi, 1000))
        result = linear_interpolation(signal)
        assert result.min() >= signal.min() - 0.01
        assert result.max() <= signal.max() + 0.01

    def test_upsampling_from_short_signal(self):
        from ecgtizer.completion import linear_interpolation
        signal = np.array([0.0, 1.0, 0.0])
        result = linear_interpolation(signal)
        assert len(result) == 5000
        # Peak should still be around 1.0
        assert result.max() >= 0.99

    def test_downsampling_from_long_signal(self):
        from ecgtizer.completion import linear_interpolation
        signal = np.random.randn(10000)
        result = linear_interpolation(signal)
        assert len(result) == 5000


class TestDenormalization:

    def test_basic_denormalization(self):
        from ecgtizer.completion import denormalization
        # Normalized value of -1 should map to original_min
        result = denormalization(np.array([-1.0]), 10.0, 20.0)
        np.testing.assert_allclose(result, [10.0])

    def test_denorm_max(self):
        from ecgtizer.completion import denormalization
        # Normalized value of 1 should map to original_max
        result = denormalization(np.array([1.0]), 10.0, 20.0)
        np.testing.assert_allclose(result, [20.0])

    def test_denorm_midpoint(self):
        from ecgtizer.completion import denormalization
        # Normalized value of 0 should map to midpoint
        result = denormalization(np.array([0.0]), 0.0, 100.0)
        np.testing.assert_allclose(result, [50.0])

    def test_denorm_array(self):
        from ecgtizer.completion import denormalization
        signal = np.array([-1.0, 0.0, 1.0])
        result = denormalization(signal, -500, 500)
        np.testing.assert_allclose(result, [-500, 0, 500])

    def test_denorm_negative_range(self):
        from ecgtizer.completion import denormalization
        result = denormalization(np.array([0.0]), -100.0, -50.0)
        np.testing.assert_allclose(result, [-75.0])


class TestNormalization2:

    def test_output_range(self):
        from ecgtizer.completion import normalization2
        signal = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        normed, mini, maxi = normalization2(signal)
        assert normed.min() == pytest.approx(-1.0)
        assert normed.max() == pytest.approx(1.0)

    def test_returns_min_max(self):
        from ecgtizer.completion import normalization2
        signal = np.array([5.0, 10.0, 15.0])
        normed, mini, maxi = normalization2(signal)
        assert mini == 5.0
        assert maxi == 15.0

    def test_constant_signal(self):
        from ecgtizer.completion import normalization2
        signal = np.ones(100) * 5.0
        # Division by zero case - should handle or produce nan/inf
        normed, mini, maxi = normalization2(signal)
        assert mini == maxi == 5.0

    def test_roundtrip_with_denormalization(self):
        from ecgtizer.completion import normalization2, denormalization
        original = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        normed, mini, maxi = normalization2(original)
        restored = denormalization(normed, mini, maxi)
        np.testing.assert_allclose(restored, original, atol=1e-10)


class TestReplaceRandom:

    def test_output_shape_12_leads(self):
        from ecgtizer.completion import replace_random
        # normalization() expects (leads_as_rows_or_iterable, 5000) — shape (12, 5000)
        array = np.random.randn(12, 5000)
        result, scale = replace_random(array)
        assert result.shape == (1, 12, 512)

    def test_output_shape_13_leads(self):
        from ecgtizer.completion import replace_random
        # 13-lead signal (3x4 format + ref)
        array = np.random.randn(13, 5000)
        result, scale = replace_random(array)
        assert result.shape == (1, 12, 512)

    def test_scale_shape(self):
        from ecgtizer.completion import replace_random
        array = np.random.randn(12, 5000)
        result, scale = replace_random(array)
        assert scale.shape[0] == 12
        assert scale.shape[1] == 2  # min, max per lead

    def test_true_data_fills_all(self):
        from ecgtizer.completion import replace_random
        array = np.random.randn(12, 5000)
        result, scale = replace_random(array, True_data=True)
        assert result.shape == (1, 12, 512)


class TestAutoencoderNet:

    @pytest.fixture(autouse=True)
    def _skip_if_no_torch(self):
        pytest.importorskip("torch")

    def test_model_instantiation(self):
        from ecgtizer.completion import Autoencoder_net
        model = Autoencoder_net('cpu')
        assert model is not None

    def test_forward_pass_shape(self):
        """Model should accept (batch, 1, 12, 512) and return (batch, 12, 512)."""
        import torch
        from ecgtizer.completion import Autoencoder_net
        model = Autoencoder_net('cpu')
        model.eval()
        x = torch.randn(1, 1, 12, 512)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 12, 512)

    def test_forward_pass_batch(self):
        """Model should handle batch size > 1."""
        import torch
        from ecgtizer.completion import Autoencoder_net
        model = Autoencoder_net('cpu')
        model.eval()
        x = torch.randn(4, 1, 12, 512)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 12, 512)

    def test_output_range_with_tanh(self):
        """Output should be bounded by tanh [-1, 1]."""
        import torch
        from ecgtizer.completion import Autoencoder_net
        model = Autoencoder_net('cpu')
        model.eval()
        x = torch.randn(1, 1, 12, 512)
        with torch.no_grad():
            out = model(x)
        assert out.min() >= -1.0
        assert out.max() <= 1.0


class TestLoadModelAndCompletion:

    @pytest.fixture(autouse=True)
    def _skip_if_no_torch(self):
        pytest.importorskip("torch")

    def test_load_model(self, model_path):
        from ecgtizer.completion import load_model
        model = load_model(model_path, 'cpu')
        assert model is not None

    def test_completion_returns_12_leads(self, model_path, sample_ecg_dict):
        from ecgtizer.completion import completion_
        result = completion_(sample_ecg_dict, model_path, 'cpu')
        assert isinstance(result, dict)
        expected_leads = {'I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'}
        assert set(result.keys()) == expected_leads

    def test_completion_output_length(self, model_path, sample_ecg_dict):
        from ecgtizer.completion import completion_
        result = completion_(sample_ecg_dict, model_path, 'cpu')
        for lead_name, lead_data in result.items():
            assert len(lead_data) == 5000

    def test_completion_with_IIc(self, model_path, sample_ecg_dict):
        """Completion should handle IIc lead and rename it to II."""
        from ecgtizer.completion import completion_
        ecg = dict(sample_ecg_dict)
        ecg['IIc'] = ecg.pop('II')
        result = completion_(ecg, model_path, 'cpu')
        assert 'II' in result
        assert 'IIc' not in result
