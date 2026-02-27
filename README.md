# ECGtizer

**Convert PDF and image ECGs into digital signals (HL7 aECG XML)**

ECGtizer is a Python library that digitizes electrocardiogram (ECG) recordings
from PDF documents and images. It extracts the waveform traces, converts them
into numerical signal arrays, and exports them in the standard HL7 aECG XML
format. A deep-learning completion module can extend partial leads (2.5 s or
5 s) to the full 10-second recording.

---

## Features

- PDF and image input (PDF, PNG, JPG, JPEG)
- Automatic ECG format detection (Classic, Wellue, Kardia, Apple Watch)
- Three extraction algorithms with different speed/accuracy trade-offs
- Noise detection and adaptive binarization (Otsu / Sauvola thresholding)
- Deep-learning lead completion (PyTorch autoencoder)
- HL7 aECG XML export
- XML-to-PDF rendering for visual verification
- Signal comparison and analysis tools (Bland-Altman, DTW alignment)
- PDF anonymization utility

---

## Architecture

```
                         ECGtizer Pipeline
 ============================================================

  PDF / Image
       |
       v
 +------------------+
 |  convert_PDF2image|   pdf2image + poppler
 +------------------+
       |
       v
 +------------------+
 | check_noise_type |   Variance analysis on image rows/columns
 +------------------+   Detects: clean / noisy / partial noise
       |
       v
 +------------------+
 |  text_extraction  |   Mask header text and annotations
 +------------------+
       |
       v
 +------------------+
 | tracks_extraction |   Horizontal/vertical variance peaks
 +------------------+   Splits image into individual ECG strips
       |
       v
 +------------------+
 |  lead_extraction  |   Binarize + extract waveform per strip
 +------------------+   Uses selected extraction method
       |                 (lazy / full / fragmented)
       v
 +------------------+
 |   lead_cutting    |   Calibrate amplitude using ref pulse
 +------------------+   Segment strips into named leads
       |                 (I, II, III, aVR, aVL, aVF, V1-V6)
       v
 +------------------+
 |    write_xml      |   HL7 aECG XML serialization
 +------------------+

  Optional:
 +------------------+
 |   completion_     |   PyTorch autoencoder extends partial
 +------------------+   leads to full 10-second recordings
```

---

## Supported ECG Formats

| Format | Layout | Leads | Source |
|--------|--------|-------|--------|
| Classic 3x4 | 4 rows, 3 columns | I, II, III, aVR, aVL, aVF, V1-V6 | Standard 12-lead printout |
| Classic 6x2 | 2 rows, 6 columns | Same 12 leads | Alternative 12-lead layout |
| Wellue | Single strip | I (or selected lead) | Wellue portable devices |
| Kardia single | Single strip | I | AliveCor Kardia single-lead |
| Kardia multi | Multiple pages | I, II, III, aVR, aVL, aVF | AliveCor Kardia 6-lead |
| Apple Watch | Single strip | I | Apple Watch ECG export |

---

## Installation

### System dependencies

ECGtizer requires **poppler** for PDF-to-image conversion:

```bash
# macOS
brew install poppler

# Ubuntu / Debian
sudo apt-get install poppler-utils

# Fedora
sudo dnf install poppler-utils
```

### Install from source

```bash
git clone https://github.com/your-org/ecgtizer.git
cd ecgtizer
pip install -e .
```

### Development install

```bash
pip install -e ".[dev]"
pre-commit install
```

---

## Quick Start

### 1. Extract ECG signals from a PDF

```python
from ecgtizer import ECGtizer

ecg = ECGtizer(
    file="path/to/ecg.pdf",
    dpi=500,
    extraction_method="fragmented",  # "lazy", "full", or "fragmented"
    verbose=True,
)

# Access the digitized leads (dict of numpy arrays)
leads = ecg.extracted_lead
print(leads.keys())  # e.g. dict_keys(['I', 'II', 'III', ...])
```

### 2. Plot the extracted signals

```python
# Plot all leads
ecg.plot()

# Plot a specific lead with a custom range
ecg.plot(lead="II", begin=0, end=2500, save="lead_II.png")

# Overlay extraction on the original image
ecg.plot_over()
```

### 3. Export to HL7 aECG XML

```python
ecg.save_xml("output/ecg_digitized.xml")
```

### 4. Complete partial leads (deep learning)

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
ecg.completion(path_model="model/Model_Completion.pth", device=device)

# Plot completed leads
ecg.plot(completion=True)
```

### 5. Compare digitized vs original signals

```python
from ecgtizer import analyse, BlandAltman, scatter_plot

# Compute DTW alignment and correlation
results = analyse(
    path_original="data/PTB-XL/Original/00121_hr.csv",
    path_digitized="data/PTB-XL/Digitized/00121_hr.xml",
)

# Bland-Altman plot
BlandAltman(results)

# Scatter plot with regression
scatter_plot(results)
```

### 6. Convert XML back to PDF

```python
from ecgtizer import xml_to_pdf

xml_to_pdf("digitized.xml", "reconstructed.pdf")
```

### Command-line usage

```bash
python ECGtizer_main.py "data/PTB-XL/PDF/00121_hr.pdf" 500 "fragmented" \
    --verbose "output/00121_hr.xml"
```

---

## Extraction Methods

| Method | Speed | Accuracy | Noise Tolerance | Description |
|--------|-------|----------|-----------------|-------------|
| `lazy` | Fast | Moderate | High | Follows the nearest lit pixel from an anchor point. Smooths signals but handles annotations well. |
| `full` | Fast | High | Moderate | Averages all lit pixel positions per column. Captures more detail but may include annotation artifacts. |
| `fragmented` | Slower | Highest | Moderate | Combines contour detection with column-wise extraction. Best fidelity for clean recordings. |

---

## Project Structure

```
ecgtizer/
    __init__.py               Public API exports
    ecgtizer.py               ECGtizer class (main entry point)
    PDF2XML.py                Core pipeline: image processing, extraction, calibration
    PDF2XML_mod.py            Plotting, XML writing, signal utilities
    extraction_functions.py   Three extraction algorithms
    completion.py             PyTorch autoencoder for lead completion
    analyses.py               Signal comparison (DTW, Bland-Altman, etc.)
    XML2PDF.py                XML-to-PDF rendering (ecg_plot class)
    anonymisation.py          PDF anonymization utility
    fonts/                    DejaVu font files for PDF rendering
model/
    Model_Completion.pth      Pre-trained completion model weights
tests/
    conftest.py               Shared test fixtures
    test_pdf2xml.py           PDF2XML unit tests
    test_pdf2xml_mod.py       Plotting and XML writing tests
    test_extraction_functions.py  Extraction algorithm tests
    test_completion.py        Completion model tests
    test_analyses.py          Analysis function tests
    test_xml2pdf.py           XML-to-PDF tests
    test_integration.py       End-to-end integration tests
Create_database/              Synthetic ECG dataset generation tools
data/
    PTB-XL/                   Sample ECG data (PDF, CSV, XML)
```

---

## API Reference

Full API documentation is available via Sphinx:

```bash
pip install -e ".[docs]"
cd docs
make html
# open _build/html/index.html
```

### Core class

| Class / Function | Module | Description |
|-----------------|--------|-------------|
| `ECGtizer` | `ecgtizer.ecgtizer` | Main class: PDF/image to digital ECG signals |

### Analysis

| Function | Module | Description |
|----------|--------|-------------|
| `analyse` | `ecgtizer.analyses` | DTW alignment and correlation metrics |
| `BlandAltman` | `ecgtizer.analyses` | Bland-Altman agreement plot |
| `scatter_plot` | `ecgtizer.analyses` | Scatter plot with linear regression |
| `overlap_plot` | `ecgtizer.analyses` | Overlay original vs digitized signals |

### I/O utilities

| Function | Module | Description |
|----------|--------|-------------|
| `xml_to_pdf` | `ecgtizer.XML2PDF` | Render HL7 aECG XML as a PDF |
| `anonymisation` | `ecgtizer.anonymisation` | Remove patient text from ECG PDFs |

---

## Testing

```bash
# Run the full test suite
pytest tests/ -v

# Run a specific module's tests
pytest tests/test_pdf2xml.py -v
```

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Install dev dependencies: `pip install -e ".[dev]"`
4. Install pre-commit hooks: `pre-commit install`
5. Run the test suite: `pytest tests/ -v`
6. Submit a pull request

### Code style

- **Formatter:** Black (line length 120)
- **Linter:** Flake8 (max line length 120)
- **Type checker:** Mypy (informational)
- **Docstrings:** NumPy style

---

## License

This project is released into the public domain under the [Unlicense](LICENSE).

---

## Authors

- **Alex Lence** — IRD (Institut de Recherche pour le Developpement)
