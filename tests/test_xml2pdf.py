"""Tests for ecgtizer/XML2PDF.py

Tests utility functions of the ecg_plot class:
- read_lead: parse lead string to values
- ticks_positions: axis tick calculation
- lead_plot_points: signal to drawable points
- iirnotch_filter: band-stop filter
- ecg_plot initialization
- Write_PDF: end-to-end PDF generation
"""
import numpy as np
import os
import pytest

# ecg_plot loads fonts with relative path "fonts/..." so we need to be in the ecgtizer/ dir
ECGTIZER_PKG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ecgtizer")


@pytest.fixture
def ecg_plot_instance():
    """Create ecg_plot from within the ecgtizer package directory (fonts relative path)."""
    from ecgtizer.XML2PDF import ecg_plot
    orig_dir = os.getcwd()
    os.chdir(ECGTIZER_PKG_DIR)
    try:
        plot = ecg_plot()
    finally:
        os.chdir(orig_dir)
    return plot


class TestEcgPlotInit:

    def test_default_initialization(self, ecg_plot_instance):
        assert ecg_plot_instance is not None
        assert ecg_plot_instance.draw is not None

    def test_has_expected_attributes(self, ecg_plot_instance):
        assert hasattr(ecg_plot_instance, 'paper_w')
        assert hasattr(ecg_plot_instance, 'paper_h')
        assert hasattr(ecg_plot_instance, 'cols')
        assert hasattr(ecg_plot_instance, 'rows')
        assert hasattr(ecg_plot_instance, 'speed')

    def test_custom_dimensions(self):
        from ecgtizer.XML2PDF import ecg_plot
        orig_dir = os.getcwd()
        os.chdir(ECGTIZER_PKG_DIR)
        try:
            plot = ecg_plot(paper_w=300, paper_h=200)
        finally:
            os.chdir(orig_dir)
        assert plot.paper_w == 300
        assert plot.paper_h == 200


class TestReadLead:
    """read_lead is a module-level function, not an instance method."""

    def test_basic_integers(self):
        from ecgtizer.XML2PDF import read_lead
        result = read_lead("1 2 3 4 5")
        assert result == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_float_values(self):
        from ecgtizer.XML2PDF import read_lead
        result = read_lead("1.5 2.5 3.5")
        assert len(result) == 3

    def test_invalid_values_become_nan(self):
        from ecgtizer.XML2PDF import read_lead
        result = read_lead("1 abc 3")
        assert np.isnan(result[1])

    def test_empty_string(self):
        from ecgtizer.XML2PDF import read_lead
        result = read_lead("")
        assert isinstance(result, list)


class TestTicksPositions:

    def test_returns_dict(self, ecg_plot_instance):
        result = ecg_plot_instance.ticks_positions(0, 100, 25)
        assert isinstance(result, dict)

    def test_has_tick_entries(self, ecg_plot_instance):
        result = ecg_plot_instance.ticks_positions(0, 100, 25)
        assert len(result) > 0

    def test_ticks_are_evenly_spaced(self, ecg_plot_instance):
        result = ecg_plot_instance.ticks_positions(0, 100, 25)
        positions = sorted(result.keys())
        if len(positions) > 2:
            diffs = np.diff(positions)
            assert np.allclose(diffs, diffs[0], atol=0.01)


class TestIirnotchFilter:

    def test_removes_target_frequency(self, ecg_plot_instance):
        fs = 500
        t = np.arange(0, 2, 1 / fs)
        sig = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 50 * t)
        filtered = ecg_plot_instance.iirnotch_filter(sig, 50, fs)
        fft_before = np.abs(np.fft.rfft(sig))
        fft_after = np.abs(np.fft.rfft(filtered))
        freq = np.fft.rfftfreq(len(t), 1 / fs)
        idx_50 = np.argmin(np.abs(freq - 50))
        assert fft_after[idx_50] < fft_before[idx_50] * 0.5

    def test_preserves_other_frequencies(self, ecg_plot_instance):
        fs = 500
        t = np.arange(0, 2, 1 / fs)
        sig = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 50 * t)
        filtered = ecg_plot_instance.iirnotch_filter(sig, 50, fs)
        fft_before = np.abs(np.fft.rfft(sig))
        fft_after = np.abs(np.fft.rfft(filtered))
        freq = np.fft.rfftfreq(len(t), 1 / fs)
        idx_10 = np.argmin(np.abs(freq - 10))
        assert fft_after[idx_10] > fft_before[idx_10] * 0.7

    def test_output_same_length(self, ecg_plot_instance):
        sig = np.random.randn(1000)
        filtered = ecg_plot_instance.iirnotch_filter(sig, 50, 500)
        assert len(filtered) == len(sig)


class TestLeadPlotPoints:

    def test_returns_flat_list_of_coords(self, ecg_plot_instance):
        """lead_plot_points returns a flat list [x1, y1, x2, y2, ...]."""
        sig = np.sin(np.linspace(0, 2 * np.pi, 500))
        points = ecg_plot_instance.lead_plot_points(sig, x_offset=0, y_offset=100, width=200, freq=500)
        assert isinstance(points, list)
        assert len(points) > 0
        # Flat list: even count (x, y pairs)
        assert len(points) % 2 == 0

    def test_x_values_non_negative(self, ecg_plot_instance):
        sig = np.sin(np.linspace(0, 2 * np.pi, 500)) * 100
        points = ecg_plot_instance.lead_plot_points(sig, x_offset=10, y_offset=100, width=200, freq=500)
        x_vals = points[0::2]  # every other element starting from 0
        assert all(x >= 0 for x in x_vals)


class TestWritePDF:
    """Write_PDF prepends cwd to path_output, so we use relative paths."""

    def test_generates_pdf_from_dict(self, sample_ecg_dict, tmp_output_dir):
        from ecgtizer.XML2PDF import Write_PDF
        outpath = os.path.join(tmp_output_dir, "test_ecg.pdf")
        relpath = os.path.relpath(outpath)
        Write_PDF(sample_ecg_dict, relpath, type_of_pdf="type2")
        assert os.path.exists(outpath)
        assert os.path.getsize(outpath) > 0

    def test_generates_pdf_type1(self, sample_ecg_dict_13, tmp_output_dir):
        from ecgtizer.XML2PDF import Write_PDF
        outpath = os.path.join(tmp_output_dir, "test_ecg_3x4.pdf")
        ecg = dict(sample_ecg_dict_13)
        ecg['IIc'] = ecg['II'].copy()
        relpath = os.path.relpath(outpath)
        Write_PDF(ecg, relpath, type_of_pdf="type1", lead_IIc=ecg['IIc'])
        assert os.path.exists(outpath)

    def test_pdf_not_empty(self, sample_ecg_dict, tmp_output_dir):
        from ecgtizer.XML2PDF import Write_PDF
        outpath = os.path.join(tmp_output_dir, "test_ecg_size.pdf")
        relpath = os.path.relpath(outpath)
        Write_PDF(sample_ecg_dict, relpath, type_of_pdf="type2")
        assert os.path.getsize(outpath) > 1000

    def test_generates_png_output(self, sample_ecg_dict, tmp_output_dir):
        from ecgtizer.XML2PDF import Write_PDF
        outpath = os.path.join(tmp_output_dir, "test_ecg.png")
        relpath = os.path.relpath(outpath)
        Write_PDF(sample_ecg_dict, relpath, type_of_pdf="type2")
        assert os.path.exists(outpath)

    def test_with_real_csv_data(self, sample_csv_ecg, tmp_output_dir):
        from ecgtizer.XML2PDF import Write_PDF
        outpath = os.path.join(tmp_output_dir, "test_real_ecg.pdf")
        relpath = os.path.relpath(outpath)
        Write_PDF(sample_csv_ecg, relpath, type_of_pdf="type2")
        assert os.path.exists(outpath)


# --- read_xml tests ---

SAMPLE_XML_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "PTB-XL", "Digitized")


class TestReadXml:

    @pytest.fixture
    def sample_xml_path(self):
        path = os.path.join(SAMPLE_XML_DIR, "00121_hr.xml")
        if not os.path.exists(path):
            pytest.skip("Sample XML data not available")
        return path

    def test_returns_dict(self, sample_xml_path):
        from ecgtizer.XML2PDF import read_xml
        result = read_xml(sample_xml_path)
        assert isinstance(result, dict)

    def test_has_standard_leads(self, sample_xml_path):
        from ecgtizer.XML2PDF import read_xml
        result = read_xml(sample_xml_path)
        for lead in ["I", "II", "III", "AVR", "AVL", "AVF"]:
            assert lead in result, f"Missing lead: {lead}"

    def test_leads_are_numpy_arrays(self, sample_xml_path):
        from ecgtizer.XML2PDF import read_xml
        result = read_xml(sample_xml_path)
        for lead_name, lead_data in result.items():
            assert isinstance(lead_data, np.ndarray), f"Lead {lead_name} is not ndarray"

    def test_lead_length_positive(self, sample_xml_path):
        from ecgtizer.XML2PDF import read_xml
        result = read_xml(sample_xml_path)
        for lead_name, lead_data in result.items():
            assert len(lead_data) > 0, f"Lead {lead_name} is empty"


# --- xml_to_pdf end-to-end tests ---


class TestXmlToPdf:

    @pytest.fixture
    def sample_xml_path(self):
        path = os.path.join(SAMPLE_XML_DIR, "00121_hr.xml")
        if not os.path.exists(path):
            pytest.skip("Sample XML data not available")
        return path

    def test_xml_to_pdf_type1(self, sample_xml_path, tmp_output_dir):
        from ecgtizer.XML2PDF import xml_to_pdf
        outpath = os.path.join(tmp_output_dir, "xml_type1.pdf")
        relpath = os.path.relpath(outpath)
        xml_to_pdf(sample_xml_path, relpath, type_of_pdf="type1")
        assert os.path.exists(outpath)
        assert os.path.getsize(outpath) > 1000

    def test_xml_to_pdf_type2(self, sample_xml_path, tmp_output_dir):
        from ecgtizer.XML2PDF import xml_to_pdf
        outpath = os.path.join(tmp_output_dir, "xml_type2.pdf")
        relpath = os.path.relpath(outpath)
        xml_to_pdf(sample_xml_path, relpath, type_of_pdf="type2")
        assert os.path.exists(outpath)
        assert os.path.getsize(outpath) > 1000

    def test_xml_to_pdf_default_type(self, sample_xml_path, tmp_output_dir):
        from ecgtizer.XML2PDF import xml_to_pdf
        outpath = os.path.join(tmp_output_dir, "xml_default.pdf")
        relpath = os.path.relpath(outpath)
        xml_to_pdf(sample_xml_path, relpath)
        assert os.path.exists(outpath)

    def test_completed_xml_to_pdf(self, tmp_output_dir):
        """Test with a completed XML file (if available)."""
        path = os.path.join(SAMPLE_XML_DIR, "00121_hr_completed.xml")
        if not os.path.exists(path):
            pytest.skip("Completed XML data not available")
        from ecgtizer.XML2PDF import xml_to_pdf
        outpath = os.path.join(tmp_output_dir, "xml_completed.pdf")
        relpath = os.path.relpath(outpath)
        xml_to_pdf(path, relpath, type_of_pdf="type1")
        assert os.path.exists(outpath)
