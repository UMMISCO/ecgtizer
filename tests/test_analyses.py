"""Tests for ecgtizer/analyses.py

Tests:
- read_lead: parse space-separated string to numeric list
- compute_slope: linear regression slope
- alignement: signal alignment using correlation
- analyse: full analysis pipeline (correlation, MSE, DTW)
"""
import numpy as np
import pytest

from ecgtizer.analyses import read_lead, compute_slope, alignement, analyse


class TestReadLead:

    def test_basic_integers(self):
        result = read_lead("1 2 3 4 5")
        assert result == [1, 2, 3, 4, 5]

    def test_float_values(self):
        result = read_lead("1.5 2.5 3.5")
        assert result == [1, 2, 3]  # int(float(...))

    def test_negative_values(self):
        result = read_lead("-100 0 100")
        assert result == [-100, 0, 100]

    def test_invalid_values_become_zero(self):
        result = read_lead("1 abc 3")
        assert result == [1, 0, 3]

    def test_single_value(self):
        result = read_lead("42")
        assert result == [42]

    def test_empty_string(self):
        result = read_lead("")
        assert result == [0]  # empty string -> ValueError -> 0

    def test_large_string(self):
        values = " ".join(str(i) for i in range(1000))
        result = read_lead(values)
        assert len(result) == 1000


class TestComputeSlope:

    def test_identical_signals(self):
        signal = np.array([1, 2, 3, 4, 5], dtype=float)
        slope = compute_slope(signal, signal)
        assert slope == pytest.approx(1.0)

    def test_doubled_signal(self):
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        y = x * 2
        slope = compute_slope(y, x)
        assert slope == pytest.approx(2.0, abs=0.1)

    def test_negatively_correlated(self):
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        y = -x
        slope = compute_slope(y, x)
        assert slope < 0

    def test_uncorrelated_near_zero(self):
        rng = np.random.RandomState(42)
        x = rng.randn(10000)
        y = rng.randn(10000)
        slope = compute_slope(y, x)
        assert abs(slope) < 0.1


class TestAlignement:

    def test_identical_signals(self):
        signal = np.sin(np.linspace(0, 4 * np.pi, 1000))
        l1, l2 = alignement(signal, signal.copy())
        assert len(l1) == len(l2)
        # Correlation should be very high
        corr = np.corrcoef(l1, l2)[0, 1]
        assert corr > 0.95

    def test_shifted_signals(self):
        """Alignment should compensate for a temporal shift."""
        t = np.linspace(0, 4 * np.pi, 2000)
        signal1 = np.sin(t)
        # Shift by 100 samples
        signal2 = np.sin(t + 0.5)
        signal2_padded = np.concatenate([np.zeros(100), signal2])
        l1, l2 = alignement(signal1, signal2_padded)
        assert len(l1) == len(l2)
        corr = np.corrcoef(l1, l2)[0, 1]
        assert corr > 0.7

    def test_different_lengths(self):
        """Should handle signals of different lengths."""
        signal1 = np.sin(np.linspace(0, 4 * np.pi, 500))
        signal2 = np.sin(np.linspace(0, 4 * np.pi, 1000))
        l1, l2 = alignement(signal1, signal2)
        assert len(l1) == len(l2)

    def test_swaps_if_lead1_longer(self):
        """If lead1 is longer, they should be swapped internally."""
        short = np.sin(np.linspace(0, 2 * np.pi, 200))
        long = np.sin(np.linspace(0, 2 * np.pi, 500))
        l1, l2 = alignement(long, short)
        assert len(l1) == len(l2)


class TestAnalyse:

    def test_returns_dict_with_keys(self, sample_ecg_dict):
        result = analyse(sample_ecg_dict, sample_ecg_dict)
        assert 'corr' in result
        assert 'mse' in result
        assert 'dtw' in result

    def test_identical_signals_high_correlation(self, sample_ecg_dict):
        result = analyse(sample_ecg_dict, sample_ecg_dict)
        for lead, corr in result['corr'].items():
            assert corr > 0.9, f"Lead {lead} correlation too low: {corr}"

    def test_identical_signals_low_mse(self, sample_ecg_dict):
        result = analyse(sample_ecg_dict, sample_ecg_dict)
        for lead, mse in result['mse'].items():
            assert mse < 0.1, f"Lead {lead} MSE too high: {mse}"

    def test_all_12_leads_present(self, sample_ecg_dict):
        result = analyse(sample_ecg_dict, sample_ecg_dict)
        expected_leads = {'I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'}
        assert set(result['corr'].keys()) == expected_leads

    def test_different_signals_lower_correlation(self):
        """Unrelated signals should have lower correlation."""
        leads = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        rng = np.random.RandomState(42)
        ecg1 = {l: rng.randn(5000) * 500 for l in leads}
        ecg2 = {l: rng.randn(5000) * 500 for l in leads}
        result = analyse(ecg1, ecg2)
        # At least some leads should have low correlation
        avg_corr = np.mean(list(result['corr'].values()))
        assert avg_corr < 0.5

    def test_with_real_csv_data(self, sample_csv_ecg):
        """Analysis on real ECG data should not crash."""
        result = analyse(sample_csv_ecg, sample_csv_ecg)
        assert all(v > 0.9 for v in result['corr'].values())
