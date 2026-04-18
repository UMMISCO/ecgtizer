import pytest
import numpy as np
import os
import cv2

# Root of the project
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "model")
SAMPLE_CSV_DIR = os.path.join(DATA_DIR, "PTB-XL", "Original")


@pytest.fixture
def sample_ecg_dict():
    """12-lead ECG dictionary with synthetic sinusoidal signals."""
    leads = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    ecg = {}
    t = np.linspace(0, 10, 5000)
    for i, lead in enumerate(leads):
        ecg[lead] = np.sin(2 * np.pi * (1 + i * 0.1) * t) * 500
    return ecg


@pytest.fixture
def sample_ecg_dict_13():
    """13-lead ECG dictionary (12 leads + reference) for 3x4 format."""
    leads = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'ref']
    ecg = {}
    t = np.linspace(0, 10, 5000)
    for i, lead in enumerate(leads):
        ecg[lead] = np.sin(2 * np.pi * (1 + i * 0.1) * t) * 500
    return ecg


@pytest.fixture
def sample_binary_image():
    """Create a synthetic binary image with a horizontal line (simulated ECG trace)."""
    h, w = 200, 1000
    img = np.zeros((h, w), dtype=np.uint8)
    # Draw a sine wave as white pixels
    for x in range(w):
        y = int(h / 2 + 30 * np.sin(2 * np.pi * x / 200))
        y = max(0, min(h - 1, y))
        img[y, x] = 255
        # Add thickness
        if y + 1 < h:
            img[y + 1, x] = 255
        if y - 1 >= 0:
            img[y - 1, x] = 255
    return img


@pytest.fixture
def sample_binary_image_with_noise():
    """Binary image with a signal line plus random noise pixels (simulating annotations)."""
    h, w = 200, 1000
    img = np.zeros((h, w), dtype=np.uint8)
    # Draw main signal
    for x in range(w):
        y = int(h / 2 + 30 * np.sin(2 * np.pi * x / 200))
        y = max(0, min(h - 1, y))
        img[y, x] = 255
        if y + 1 < h:
            img[y + 1, x] = 255
    # Add some noise pixels at the top (simulating text)
    rng = np.random.RandomState(42)
    for x in range(100, 300):
        for y in range(10, 30):
            if rng.random() > 0.5:
                img[y, x] = 255
    return img


@pytest.fixture
def sample_color_image():
    """Create a synthetic 3-channel color image with ECG-like tracks."""
    h, w = 600, 800
    img = np.ones((h, w, 3), dtype=np.uint8) * 255
    # Draw 3 dark horizontal bands (simulated ECG tracks)
    for track_idx in range(3):
        y_center = 100 + track_idx * 200
        for x in range(50, 750):
            y = int(y_center + 20 * np.sin(2 * np.pi * x / 150))
            y = max(0, min(h - 1, y))
            img[y, x] = [0, 0, 0]
            if y + 1 < h:
                img[y + 1, x] = [0, 0, 0]
    return img


@pytest.fixture
def sample_csv_path():
    """Path to a real PTB-XL CSV sample file."""
    path = os.path.join(SAMPLE_CSV_DIR, "00129_hr.csv")
    if not os.path.exists(path):
        pytest.skip("Sample CSV data not available")
    return path


@pytest.fixture
def sample_csv_ecg(sample_csv_path):
    """Load a real ECG from PTB-XL CSV file as a dictionary."""
    import pandas as pd
    df = pd.read_csv(sample_csv_path)
    lead_map = {
        'I': 'I', 'II': 'II', 'III': 'III',
        'aVR': 'AVR', 'aVL': 'AVL', 'aVF': 'AVF',
        'V1': 'V1', 'V2': 'V2', 'V3': 'V3',
        'V4': 'V4', 'V5': 'V5', 'V6': 'V6'
    }
    ecg = {}
    for csv_name, std_name in lead_map.items():
        ecg[std_name] = df[csv_name].values * 1000  # Convert to µV
    return ecg


@pytest.fixture
def model_path():
    """Path to the pre-trained completion model."""
    path = os.path.join(MODEL_DIR, "Model_Completion.pth")
    if not os.path.exists(path):
        pytest.skip("Model file not available")
    return path


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Temporary directory for test outputs."""
    return str(tmp_path)
