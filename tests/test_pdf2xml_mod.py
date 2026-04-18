"""Tests for ecgtizer/PDF2XML_mod.py

Tests utility functions:
- transform_np2txt: numpy array to space-separated string
- conversion_time: date/time formatting
- write_xml: XML file writing (integration)
- plot_function: plot generation (integration)
"""
import numpy as np
import os
import pytest
import xml.etree.ElementTree as ET

from ecgtizer.PDF2XML_mod import (
    transform_np2txt,
    conversion_time,
    write_xml,
    plot_function,
)


class TestTransformNp2Txt:

    def test_basic_integer_array(self):
        arr = np.array([1, 2, 3, 4, 5])
        result = transform_np2txt(arr)
        assert result == "1 2 3 4 5"

    def test_float_array(self):
        arr = np.array([1.5, 2.5, 3.5])
        result = transform_np2txt(arr)
        assert result == "1.5 2.5 3.5"

    def test_single_element(self):
        arr = np.array([42])
        result = transform_np2txt(arr)
        assert result == "42"

    def test_negative_values(self):
        arr = np.array([-1, 0, 1])
        result = transform_np2txt(arr)
        assert result == "-1 0 1"

    def test_no_trailing_space(self):
        arr = np.array([10, 20, 30])
        result = transform_np2txt(arr)
        assert not result.endswith(" ")

    def test_no_leading_space(self):
        arr = np.array([10, 20, 30])
        result = transform_np2txt(arr)
        assert not result.startswith(" ")

    def test_large_array(self):
        arr = np.arange(1000)
        result = transform_np2txt(arr)
        parts = result.split(" ")
        assert len(parts) == 1000


class TestConversionTime:

    def test_known_date(self):
        result = conversion_time('15', 'Mar', '2023', '14:30')
        assert result == '202303151430'

    def test_all_unknown(self):
        result = conversion_time('unknow', 'unknow', 'unknow', 'unknow')
        assert result == '000000000000'

    def test_unknown_hour(self):
        result = conversion_time('01', 'Jan', '2024', 'unknow')
        assert result == '202401010000'

    def test_unknown_day(self):
        result = conversion_time('unknow', 'Feb', '2024', '10:00')
        assert result == '202402001000'

    def test_unknown_year(self):
        result = conversion_time('25', 'Dec', 'unknow', '23:59')
        assert result == '000012252359'

    def test_all_months(self):
        months = {
            'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
            'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
            'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
        }
        for month_str, month_num in months.items():
            result = conversion_time('01', month_str, '2024', '00:00')
            assert month_num in result


class TestWriteXml:

    def test_writes_file(self, sample_ecg_dict, tmp_output_dir):
        """write_xml should create an XML file on disk."""
        outpath = os.path.join(tmp_output_dir, "test_output.xml")
        table = {
            'low_freq': '0.05', 'high_freq': '150', 'BPM': '72',
            'Inter PR (ms)': '160', 'Dur.QRS (ms)': '100',
            'QT (ms)': '400', 'QTc (ms)': '420',
            'Axe P': '60', 'Axe R': '30', 'Axe T': '45',
            'Moy RR (ms)': '833', 'QTcB (ms)': '415', 'QTcF (ms)': '410',
            'Rythme': 'Sinus', 'ECG': 'Normal',
            'Age': '50', 'sex': 'M', 'other_information': 'None'
        }
        write_xml(sample_ecg_dict, outpath, TYPE='classic', table=table)
        assert os.path.exists(outpath)

    def test_xml_is_valid(self, sample_ecg_dict, tmp_output_dir):
        """Written XML should be parseable."""
        outpath = os.path.join(tmp_output_dir, "test_valid.xml")
        table = {
            'low_freq': '0.05', 'high_freq': '150', 'BPM': '72',
            'Inter PR (ms)': '160', 'Dur.QRS (ms)': '100',
            'QT (ms)': '400', 'QTc (ms)': '420',
            'Axe P': '60', 'Axe R': '30', 'Axe T': '45',
            'Moy RR (ms)': '833', 'QTcB (ms)': '415', 'QTcF (ms)': '410',
            'Rythme': 'Sinus', 'ECG': 'Normal',
            'Age': '50', 'sex': 'M', 'other_information': 'None'
        }
        write_xml(sample_ecg_dict, outpath, TYPE='classic', table=table)
        tree = ET.parse(outpath)
        root = tree.getroot()
        assert root.tag.endswith('AnnotatedECG')

    def test_xml_contains_leads(self, sample_ecg_dict, tmp_output_dir):
        """XML should contain component elements for each lead."""
        outpath = os.path.join(tmp_output_dir, "test_leads.xml")
        table = {
            'low_freq': '0.05', 'high_freq': '150', 'BPM': '72',
            'Inter PR (ms)': '160', 'Dur.QRS (ms)': '100',
            'QT (ms)': '400', 'QTc (ms)': '420',
            'Axe P': '60', 'Axe R': '30', 'Axe T': '45',
            'Moy RR (ms)': '833', 'QTcB (ms)': '415', 'QTcF (ms)': '410',
            'Rythme': 'Sinus', 'ECG': 'Normal',
            'Age': '50', 'sex': 'M', 'other_information': 'None'
        }
        write_xml(sample_ecg_dict, outpath, TYPE='classic', table=table)
        tree = ET.parse(outpath)
        xml_str = ET.tostring(tree.getroot(), encoding='unicode')
        # Should contain digit data for leads
        assert 'digits' in xml_str

    def test_creates_output_directory(self, sample_ecg_dict, tmp_output_dir):
        """Should create intermediate directories if they don't exist."""
        outpath = os.path.join(tmp_output_dir, "subdir", "deep", "test.xml")
        table = {
            'low_freq': '0.05', 'high_freq': '150', 'BPM': '72',
            'Inter PR (ms)': '160', 'Dur.QRS (ms)': '100',
            'QT (ms)': '400', 'QTc (ms)': '420',
            'Axe P': '60', 'Axe R': '30', 'Axe T': '45',
            'Moy RR (ms)': '833', 'QTcB (ms)': '415', 'QTcF (ms)': '410',
            'Rythme': 'Sinus', 'ECG': 'Normal',
            'Age': '50', 'sex': 'M', 'other_information': 'None'
        }
        write_xml(sample_ecg_dict, outpath, TYPE='classic', table=table)
        assert os.path.exists(outpath)


class TestPlotFunction:

    def test_single_lead_plot_does_not_crash(self, sample_ecg_dict):
        """Plotting a single lead should not raise."""
        import matplotlib
        matplotlib.use('Agg')
        plot_function(sample_ecg_dict, lead='I')

    def test_multilead_12_plot_does_not_crash(self, sample_ecg_dict):
        """Plotting 12-lead grid should not raise."""
        import matplotlib
        matplotlib.use('Agg')
        plot_function(sample_ecg_dict)

    def test_multilead_13_plot_does_not_crash(self, sample_ecg_dict_13):
        """Plotting 13-lead (3x4) grid should not raise."""
        import matplotlib
        matplotlib.use('Agg')
        plot_function(sample_ecg_dict_13)

    def test_save_plot_to_file(self, sample_ecg_dict, tmp_output_dir):
        """Saving a plot should create a file."""
        import matplotlib
        matplotlib.use('Agg')
        outpath = os.path.join(tmp_output_dir, "test_plot.png")
        plot_function(sample_ecg_dict, save=outpath)
        assert os.path.exists(outpath)
