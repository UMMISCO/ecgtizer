"""Core image-processing pipeline for ECG digitization.

Converts PDF/image ECGs into numerical signal arrays through noise detection,
adaptive binarization, track segmentation, waveform extraction and amplitude
calibration using a reference pulse.
"""

from __future__ import annotations

import logging

import numpy as np
from pdf2image import convert_from_path, exceptions
from .extraction_functions import lazy_extraction, full_extraction, fragmented_extraction
import cv2
from scipy import signal
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# --- Signal parameters ---
SAMPLING_FREQ = 500  # Hz
SIGNAL_LENGTH_STANDARD = 5000  # 10 seconds at 500 Hz (Wellue/other)
SIGNAL_LENGTH_CLASSIC = 5140  # Classic format signal length
SIGNAL_LENGTH_KARDIA = 4000  # Kardia format signal length
AMPLITUDE_SCALE_UV = 1000  # Scaling factor for µV conversion

# --- Reference pulse lengths (in samples) ---
REF_PULSE_APPLE = 180
REF_PULSE_KARDIA = 240
REF_PULSE_GENERIC = 300
REF_PULSE_CLASSIC = 330

# --- Image noise/variance thresholds ---
VARIANCE_NOISY = 3000  # Above this → image is noisy
VARIANCE_HIGH = 2000  # Above this or below LOW → might be noisy
VARIANCE_LOW = 600  # Below HIGH or above this → might be noisy
NOISE_PARTIAL = 0.5  # Intermediate noise state

# --- Image processing thresholds ---
LINE_VARIANCE_MIN = 1000  # Min row variance to keep during image cleanup
COLUMN_VARIANCE_MIN = 200  # Min column variance to keep during image cleanup
WAVEFORM_VARIANCE_MIN = 200  # Min vertical variance to detect signal presence

# --- Pixel values ---
WHITE_PIXEL = 255

# --- PDF rasterization safety caps (decompression-bomb defense) ---
MAX_PDF_PAGES = 5  # ECG printouts are single-page; allow margin
MAX_DPI = 1200  # 2.4x typical 500 DPI; refuse pathological values

# --- Lead timing boundaries (samples) ---
LEAD_TIME_3X4 = {
    "I": (0, 1250),
    "II": (0, 1250),
    "III": (0, 1250),
    "AVR": (1250, 2500),
    "AVL": (1250, 2500),
    "AVF": (1250, 2500),
    "V1": (2500, 3750),
    "V2": (2500, 3750),
    "V3": (2500, 3750),
    "V4": (3750, 5000),
    "V5": (3750, 5000),
    "V6": (3750, 5000),
    "IIc": (0, 5000),
}
LEAD_TIME_6X2 = {
    "I": (0, 2500),
    "II": (0, 2500),
    "III": (0, 2500),
    "AVR": (0, 2500),
    "AVL": (0, 2500),
    "AVF": (0, 2500),
    "V1": (2500, 5000),
    "V2": (2500, 5000),
    "V3": (2500, 5000),
    "V4": (2500, 5000),
    "V5": (2500, 5000),
    "V6": (2500, 5000),
}


def _binarize_image(image: np.ndarray, TYPE: str, NOISE: bool | float) -> np.ndarray:
    """Binarize a grayscale or BGR image using the appropriate thresholding method.

    For clean colour images (NOISE=False, not Wellue), Otsu thresholding is
    followed by a colour-based filter that removes ECG grid lines.  Standard
    ECG paper has an orange/pink grid whose brightest RGB channel is well above
    128, whereas the black trace has max channel below 50.  Discarding lit
    pixels with ``max(R,G,B) > 128`` cleanly separates trace from grid.
    """
    if image.ndim == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = image
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    if NOISE:
        _, image_bin = cv2.threshold(img_gray, 40, 255, cv2.THRESH_BINARY_INV)
    elif TYPE.lower() == "wellue":
        _, image_bin = cv2.threshold(img_blur, 127, 255, cv2.THRESH_BINARY_INV)
    else:
        _, image_bin = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Remove coloured grid lines: on standard ECG paper the grid is
        # orange/pink (max channel ~150-240) while the trace is black
        # (max channel ~0-50).  Any lit pixel whose brightest channel
        # exceeds 128 is grid, not trace.
        if image.ndim == 3:
            max_channel = np.max(image, axis=2)
            image_bin[max_channel > 128] = 0
    return image_bin


def _calibrate_from_binary(track_bin: np.ndarray, DPI: int = 0) -> tuple[float, float]:
    """Measure the calibration square directly from a binary track image.

    Examines the first 15 % of the track width (the reference pulse region)
    and finds columns where the lit-pixel span is significantly larger than
    the median trace thickness.  Uses a two-stage detection: first finds the
    maximum span (the vertical edges of the calibration square), then selects
    only columns near that maximum to compute the calibration height.

    Returns (pixel_zero, factor).  ``pixel_zero`` is the baseline row
    (bottom of the square) and ``factor`` is the square height in pixels,
    corresponding to 1 mV on the ECG.  Falls back to a DPI-based estimate
    when the square cannot be detected.
    """
    h, w = track_bin.shape
    ref_width = max(10, int(0.15 * w))
    ref_region = track_bin[:, :ref_width]

    # Per-column: lit pixel span (max_row - min_row)
    spans = np.zeros(ref_width)
    baselines = np.zeros(ref_width)
    for c in range(ref_width):
        lit = np.where(ref_region[:, c] == 255)[0]
        if len(lit) >= 2:
            spans[c] = lit[-1] - lit[0]
            baselines[c] = float(lit[-1])  # bottom-most lit pixel

    # The calibration square has a much larger span than the thin trace
    valid = spans > 0
    if np.sum(valid) < 3:
        # Not enough data — DPI fallback
        f = (10 * DPI) / 25.4 if DPI > 0 else 1.0
        pixel_zero = float(h / 2)
        return pixel_zero, f

    max_span = float(np.max(spans))
    median_span = float(np.median(spans[valid]))

    # Two-stage detection:
    # 1) The max span represents the vertical edge of the calibration square.
    #    If it's much larger than the median trace (> 5×) and plausible
    #    (< 50% of track height — the square is ~10mm on ~50mm tracks),
    #    use columns near that maximum (> 50% of max_span) as the true
    #    square columns.
    # 2) Fall back to the original 3× median threshold otherwise.
    if max_span > median_span * 5 and max_span > 10 and max_span < h * 0.5:
        edge_mask = spans > max_span * 0.5
        if np.sum(edge_mask) >= 2:
            f = float(np.median(spans[edge_mask]))
            pixel_zero = float(np.median(baselines[edge_mask]))
            if f >= 1:
                return pixel_zero, f

    # Fallback: original threshold-based detection
    square_mask = spans > max(median_span * 3, 10)
    if np.sum(square_mask) < 3:
        f = (10 * DPI) / 25.4 if DPI > 0 else 1.0
        pixel_zero = float(np.median(baselines[valid]))
        return pixel_zero, f

    f = float(np.median(spans[square_mask]))
    pixel_zero = float(np.median(baselines[square_mask]))

    if f < 1:
        f = (10 * DPI) / 25.4 if DPI > 0 else 1.0
    return pixel_zero, f


def _calibrate_ref_pulse(ref_pulse: np.ndarray, DPI: int = 0) -> tuple[float, float]:
    """Compute calibration (pixel_zero, factor) from a reference pulse segment.

    Returns (pixel_zero, factor) where factor converts pixel distance to µV.
    """
    pixel_zero = float(max(ref_pulse[:10]) if len(ref_pulse) >= 10 else max(ref_pulse))
    pixel_one = float(min(ref_pulse))

    # Flat pulse fallback
    if np.all(np.diff(ref_pulse) == 0):
        pixel_zero = float(np.mean(ref_pulse))
        pixel_one = float(min(ref_pulse))

    f = pixel_zero - pixel_one
    if f == 0:
        if DPI > 0:
            f = (10 * DPI) / 25.4
        else:
            f = 1.0
    return pixel_zero, f


def convert_PDF2image(path_input: str, DPI: int) -> np.ndarray:
    """
    Convert the PDF file into array (images).

    We use the library pdf2image to transform the input file into an array

    Parameters
    ----------
    path_input : str, path of the pdf file to convert
    DPI :int, dots per inch (resolution of the image)

    Returns
    -------
    list : list of all the pages of the PDF in PIL format
    int  : number of pages
    bool : True: The conversion has worked / False :  The conversion has not worked
    """
    if DPI > MAX_DPI:
        logger.error("DPI %d exceeds MAX_DPI=%d; refusing to rasterize.", DPI, MAX_DPI)
        return ("_", "_", False)
    try:
        pages = convert_from_path(path_input, dpi=DPI, first_page=1, last_page=MAX_PDF_PAGES)
    except exceptions.PDFPageCountError:
        logger.error("Impossible conversion. The input file is not a PDF.")
        return ("_", "_", False)
    return (pages, len(pages), True)


def check_noise_type(image: np.ndarray, DPI: int, DEBUG: bool) -> tuple[str, bool | float]:
    """
    Check the noise level of the image. Check the type of the image.

    Parameters
    ----------
    image : np.array, image
    DPI   : int, dots per inch (resolution of the image)
    DEBUG : bool, show the image

    Returns
    -------
    str : Type of image
    bool : True: The image is noised / False : The image is not noised
    """

    # Check color diversity along the middle column (vectorized)
    mid_col = image[:, image.shape[1] // 2, :]  # shape: (H, 3)
    # Kardia: only one unique green-channel value in rows where R==255 or B is truthy
    mask = (mid_col[:, 0] == 255) | (mid_col[:, 2] != 0)
    unique_colors = np.unique(mid_col[mask, 1]) if np.any(mask) else np.array([])
    if len(unique_colors) <= 1:
        return ("Kardia", False)

    # Check the variance in the image (compute once)
    image_var = np.var(image)
    if image_var > VARIANCE_HIGH or image_var < VARIANCE_LOW:
        if image_var > VARIANCE_NOISY:
            NOISE = True
        else:
            NOISE = NOISE_PARTIAL
    else:
        NOISE = False

    if len(image) > len(image[0]):
        # the Wellue format offers images that are taller than they are wide
        return ("Wellue", NOISE)

    else:
        # Convert image in gray scale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Binarize the image
        ret, thresh1 = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        # Define the rectangle original size
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(0.03 * len(image)), int(0.03 * len(image))))
        # Dilate the image
        dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)
        # Find contour by applying rectangle
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        im2 = image.copy()
        nbr = 0
        # Count the number of rectangles in the apple watch format there is 3 record rectangle
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w - x > len(image) / 3:
                rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (255, 0, 0), 2)
                nbr += 1
        # Plot the image with the different rectangle(s) find
        if DEBUG:
            try:
                plt.figure(figsize=(20, 14))
                plt.imshow(rect)
                plt.show()
            except UnboundLocalError:
                pass
        # There are more than 3 record rectangle it is apple watch format
        if nbr >= 3:
            return ("apple", False)
        # There is less than 3 record rectangle it is a classical format
        else:
            return ("classic", NOISE)


def text_extraction(
    image: np.ndarray, page: int, DPI: int, NOISE: bool | float, TYPE: str, DEBUG: bool
) -> tuple[dict, np.ndarray, str, str]:
    """
    Extract the texte from the image and mask the task on the image
    For Kardia it mask the gride line

    Parameters
    ----------
    image : np.array, image
    DPI   : int, dots per inch (resolution of the image)
    NOISE : bool, if the image is noised or not
    TYPE  : str, format of the image
    DEBUG : bool, show the image

    Returns
    -------
    array : The image without the text
    DataFrame : The dataframe with the extracted text in it
    """
    df = []

    if TYPE.lower() == "kardia":
        # Isolate the record region
        work_image = np.array(image)[DPI : int(10 * DPI), int(0.3 * DPI) : int(8 * DPI)]
        # Convert the image in gray scale
        image_gray = cv2.cvtColor(work_image, cv2.COLOR_BGR2GRAY)
        # Binarize the image thanks to the gray scale
        new_image = np.where(image_gray == 0, WHITE_PIXEL, 0).astype(image_gray.dtype)

        # Compute the vertical variance
        var_line = np.var(new_image, axis=1)
        # Compute the horizontal variance
        var_column = np.var(new_image, axis=0)
        # Define a second image to work with
        working_image = np.copy(new_image)

        for i in range(len(new_image)):
            if var_line[i] < LINE_VARIANCE_MIN:
                working_image[i, :] = 0
        for i in range(len(new_image[0])):
            if var_column[i] < COLUMN_VARIANCE_MIN:
                working_image[:, i] = 0

        if DEBUG:
            plt.figure(figsize=(20, 14))
            plt.imshow(working_image)
            plt.show()
        return (working_image, df)

    # Table with the information patient in it

    # Convert image in gray scale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply a Gaussian Blur
    image_blur = cv2.GaussianBlur(image_gray, (5, 5), 0)
    # If the image is noised we apply a deterministic threshold
    if NOISE:
        # Binarize the image with the deterministic threshold
        ret, image_bin = cv2.threshold(image_gray, 40, 100, cv2.THRESH_BINARY_INV)
        # Compute the horizontal variance
        horizontal_variance = np.var(image_bin, axis=1)
        # Detect the variance peaks
        peaks = signal.argrelextrema(horizontal_variance, np.greater, order=int(len(image) / 10))[
            0
        ]  # Compute the pikes position
        # Mask the text region if peaks were found
        if len(peaks) >= 2:
            # starting position on the x-axis
            x = 0
            # Ending position on the x-axis
            w = len(image[0])
            # Starting position on the y-axis
            y = 0
            # Ending position on the y-axis
            h = int(peaks[0] + (peaks[1] - peaks[0]) / 2)
            im2 = image.copy()
            # Define and apply a mask on the text region
            rect = cv2.rectangle(image_bin, (x, peaks[0]), (x + w, y + h), (255, 0, 0), 2)
            # The mask must have the same color as the rest of the image
            image[y : y + h, x : x + w] = np.mean(image[y : y + h, x : x + w])

    # If the image is not noised we apply a Otsu detection threshold
    else:
        # Binarize the image with the Otsu threshold
        ret, image_bin = cv2.threshold(image_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        img_h, img_w = image.shape[:2]

        # ── Pass 1: Contour-based removal of medium text blocks ──
        if TYPE == "apple":
            k_size = int(0.03 * img_h)
        else:
            k_size = max(int(0.0075 * img_h), 8)
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
        dilation = cv2.dilate(image_bin, rect_kernel, iterations=1)
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        im2 = image.copy()

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area_ratio = (w * h) / (img_w * img_h)
            if area_ratio > 0.25:
                continue
            if img_w > img_h:  # landscape
                if w < img_w / 3:
                    rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    image[y : y + h, x : x + w] = (255, 255, 255)
            else:  # portrait
                if h < img_h / 4:
                    rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    image[y : y + h, x : x + w] = np.mean(image[y : y + h, x : x + w])

        # ── Passes 2 & 3 apply only to low-res images (photos). ──
        # High-res PDFs have clean contours and don't need aggressive
        # character-level removal — Pass 1 is sufficient for them.
        # Threshold: ~2 megapixels separates photos from 500-DPI PDFs.
        is_lowres = (img_h * img_w) < 2_000_000

        if is_lowres:
            # ── Pass 2: Zone-based header / footer removal ──
            # In standard ECG printouts, text lives above the first
            # track and below the last track.  Use the horizontal
            # projection of the binary image to find the signal band
            # and blank everything outside it.
            row_proj = np.sum(image_bin, axis=1).astype(float)
            from scipy.ndimage import uniform_filter1d

            row_proj_smooth = uniform_filter1d(row_proj, size=max(img_h // 40, 5))
            proj_thresh = row_proj_smooth.max() * 0.05
            active_rows = np.where(row_proj_smooth > proj_thresh)[0]

            if len(active_rows) > 2:
                first_active = int(active_rows[0])
                last_active = int(active_rows[-1])
                # Header: blank above first active row
                header_end = max(0, first_active - max(int(img_h * 0.005), 2))
                if header_end > int(img_h * 0.02):
                    image[:header_end, :] = (255, 255, 255) if image.ndim == 3 else 255
                # Footer: blank below last active row
                footer_start = min(img_h, last_active + max(int(img_h * 0.005), 2))
                if (img_h - footer_start) > int(img_h * 0.02):
                    image[footer_start:, :] = (255, 255, 255) if image.ndim == 3 else 255

            # NOTE: A character-level pass (connected-component or
            # morphological) was tested here but removed — it
            # reliably destroys calibration squares on photos because
            # those squares are similar in size/shape to text blobs.
            # Lead labels within the signal area remain; they are
            # handled as noise during waveform extraction.

    # Plot the image with the detected rectangles
    if DEBUG:
        try:
            plt.figure(figsize=(20, 14))
            plt.imshow(rect)
            plt.show()
            plt.imshow(image)
            plt.show()
        except UnboundLocalError:
            plt.imshow(image)
            plt.show()
    return image


def tracks_extraction(
    image: np.ndarray, TYPE: str, DPI: int, FORMAT: str, NOISE: bool | float = False, DEBUG: bool = False
) -> dict[int, np.ndarray]:
    """
    Extract the tracks from the image

    Parameters
    ----------
    image : np.array, image
    TYPE  : str, format of the image
    DPI   : int, dots per inch (resolution of the image)
    FORMAT: str, multi or unilead for Kardia
    NOISE : bool, if the image is noised or not
    DEBUG : bool, show the image

    Returns
    -------
    dictionary  : dictionary of the different extracted tracks with their position
                  (key : position / Value: Track images)
    """
    # dictionary of all tracks
    dic_tracks = {}
    if TYPE.lower() == "kardia":
        var_line = np.var(image, axis=1)
        peaks, _ = find_peaks(var_line, height=2 * DPI, distance=DPI)
        start = 0
        it = 0
        for p in range(len(peaks) - 1):
            end = (peaks[p] + peaks[p + 1]) / 2
            dic_tracks[it] = image[start : int(end), :]
            start = int(end)
            it += 1
        dic_tracks[it] = image[start:, :]

        dic_tracks_temp = {}
        it = 0
        if FORMAT == "unilead":
            for i in dic_tracks:
                if i % 2 == 0:
                    dic_tracks_temp[it] = dic_tracks[i]
                    it += 1
            if DEBUG:
                for im in dic_tracks_temp:
                    plt.imshow(dic_tracks_temp[im])
                    plt.show()
            return dic_tracks_temp

        else:
            if DEBUG:
                for im in dic_tracks:
                    plt.imshow(dic_tracks[im])
                    plt.show()
            return dic_tracks

    # Plot the original image
    if DEBUG:
        plt.figure(figsize=(20, 14))

    # Binarize the image using the appropriate thresholding method
    image_bin = _binarize_image(image, TYPE, NOISE)

    # Compute the horizontal variance on binarized image
    horizontal_variance = np.var(image_bin, axis=1)

    # If images are taller than they are wide the distance between two peaks is smaller
    if len(image) > len(image[0]):
        # Compute the pikes position
        peaksh = signal.argrelextrema(horizontal_variance, np.greater, order=int(0.010 * len(image)))[0]

    # If images are wide than they are taller the distance between two peaks is bigger
    if len(image) < len(image[0]):
        # Compute the pikes position
        # peaks = signal.argrelextrema(horizontal_variance, np.greater, order = int(0.05*len(image)))[0]
        peaksh, _ = find_peaks(horizontal_variance, height=len(image[0]), distance=int(len(image) / 10))
        # if NOISE:
        #     peaksh, _ = find_peaks(horizontal_variance, height=(len(image)-(len(image[0])*15/100),len(image[0])), distance=int(len(image)/10))

    if DEBUG:
        plt.plot(horizontal_variance)
        for p in peaksh:
            plt.axvline(p, c="r")
        plt.savefig("Horrizontal_variance.png")
        plt.show()
        plt.figure(figsize=(20, 20))

    # Define a list with all the position to cut between tracks
    cut_pos = []
    if len(peaksh) >= 2:
        # Use the typical inter-peak spacing to estimate where track 0
        # starts.  The first cut should be half a track above the first
        # peak rather than at row 0, which would include blank header.
        typical_gap = int(np.median(np.diff(peaksh)))
        first_cut = max(0, peaksh[0] - typical_gap // 2)
        cut_pos.append(first_cut)
        for p in range(len(peaksh) - 1):
            cut_pos.append(int((peaksh[p] + peaksh[p + 1]) / 2))
        last_cut = min(len(image), peaksh[-1] + typical_gap // 2)
        cut_pos.append(last_cut)
    else:
        # Fallback: use old behaviour
        cut_pos = [0]
        for p in range(len(peaksh) - 1):
            cut_pos.append(int((peaksh[p] + peaksh[p + 1]) / 2))
        cut_pos.append(len(image))

    # If we have 6 cuts we have extracted text information
    if len(cut_pos) == 6:
        del cut_pos[0]

    # We store all track image in the dictionary
    it = 1
    for c in range(len(cut_pos) - 1):
        if it == 1:
            dic_tracks[c] = image_bin[cut_pos[c] + int(0.02 * len(image)) : cut_pos[c + 1]]
        elif it == len(cut_pos) - 1:
            dic_tracks[c] = image_bin[cut_pos[c] : cut_pos[c + 1] - int(0.02 * len(image))]
        else:
            dic_tracks[c] = image_bin[cut_pos[c] : cut_pos[c + 1]]

        it += 1

        if DEBUG:
            plt.axhline(cut_pos[c], c="g", alpha=0.6)

    # Plot the position of the cut in the image
    if DEBUG:
        plt.imshow(image)
        plt.axhline(cut_pos[-1], c="g", alpha=0.6)
        for p in peaksh:
            plt.axhline(p, c="r", alpha=0.6)

    # Compute the vertical variance
    vertical_variance = np.var(image_bin, axis=0)

    # Find positions where variance indicates signal waveform presence
    peaksv = np.where(vertical_variance > WAVEFORM_VARIANCE_MIN)[0].tolist()

    # For all the tracks we cut vertically the part which not contain waveform
    if peaksv:
        v_start, v_end = peaksv[0], peaksv[-1]
        for track in dic_tracks.keys():
            dic_tracks[track] = dic_tracks[track][:, v_start:v_end]
    else:
        v_start = 0

    # Plot the position of the cut in the image
    if DEBUG and peaksv:
        plt.axvline(peaksv[0])
        plt.axvline(peaksv[-1])
        plt.savefig("Image_of_tracks.png")
        plt.show()

        plt.plot(vertical_variance)
        plt.axvline(peaksv[0], c="r")
        plt.axvline(peaksv[-1], c="r")
        plt.savefig("Vertical_variance.png")
        plt.show()

    return (dic_tracks, peaksh, v_start)


def clean_tracks(
    dic_tracks: dict[int, np.ndarray], TYPE: str, NOISE: bool | float, DEBUG: bool
) -> dict[int, np.ndarray]:
    """
    Detect all groups of pixels and remove them

    Parameters
    ----------
    dic_tracks: dictionary, dictionary of track images
    TYPE  : str, format of the image
    NOISE : bool, if the image is noised or not
    DEBUG : bool, show the image

    Returns
    -------
    None
    """

    for d in dic_tracks:
        # Kardia files are already binarize
        if TYPE.lower() != "kardia":
            # Convert the image in gray scale
            img_gray = cv2.cvtColor(dic_tracks[d], cv2.COLOR_BGR2GRAY)
            # Apply a Gaussian Blur
            img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

            # If the image is Wellue type we have determine the optimal threshold
            if TYPE == "Wellue":
                ret, image_bin = cv2.threshold(img_blur, 127, 255, cv2.THRESH_BINARY_INV)

            else:
                # For noisy images, use adaptive (local) thresholding so the
                # decision follows local illumination instead of one global
                # threshold. Previously this branch called skimage's
                # threshold_sauvola without importing it — a latent NameError.
                # cv2.adaptiveThreshold gives equivalent local-window behaviour
                # without pulling in scikit-image.
                if NOISE is True:
                    image_bin = cv2.adaptiveThreshold(
                        img_blur, WHITE_PIXEL,
                        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                        blockSize=11, C=2,
                    )
                else:
                    ret, image_bin = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            # Define the rectangle original size
            rect_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (int(0.045 * len(dic_tracks[d])), int(0.045 * len(dic_tracks[d])))
            )

        else:
            image_bin = dic_tracks[d].astype("uint8")
            # Define the rectangle original size
            rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))

        # Dilate the image
        dilation = cv2.dilate(image_bin, rect_kernel, iterations=1)
        # Find contour by applying rectangle
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        im2 = dic_tracks[d].copy()

        # For all the rectangles with a certain size mask them
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w - x < 100 and h - y < 100:
                rect = cv2.rectangle(im2.astype("uint8"), (x, y), (x + w, y + h), (255, 0, 0), 2)
                dic_tracks[d][y : y + h, x : x + w] = np.mean(dic_tracks[d][y : y + h, x : x + w])

        # Plot the image and the associated masks
        if DEBUG:
            plt.figure(figsize=(20, 14))
            try:
                plt.imshow(rect)
            except UnboundLocalError:
                plt.imshow(im2)
            plt.show()


def sup_holes(signal: list | np.ndarray, TYPE: str) -> np.ndarray:
    """
    Fill the holes in the extracted signal

    Parameters
    ----------
    signal: array, contain the extracted signal
    TYPE: str, it can be :
            - "classic"
            - "heartcheck"
            - "duoek"

    Returns
    -------
    list: list of the extracted signal without hole
    """

    # if the signal is constant then we set the signal to 0
    signal = np.asarray(signal, dtype=float)
    if np.all(np.diff(signal) == 0):
        signal = np.zeros(len(signal))
        return signal

    end = -1
    # Treat both zeros and NaN as holes to interpolate
    hole_mask = (signal == 0) | np.isnan(signal)

    # If the first value is a hole, find the next valid point
    if hole_mask[0]:
        j = 1
        while j < len(signal) and hole_mask[j]:
            j += 1
        if j < len(signal):
            signal[0] = signal[j]
        else:
            signal[0] = len(signal) / 2  # fallback to midpoint

    # If the last value is a hole, find the previous valid point
    if hole_mask[-1]:
        j = 1
        while j < len(signal) and hole_mask[-j]:
            j += 1
        if j < len(signal):
            signal[-1] = signal[-j]
        else:
            signal[-1] = signal[0]

    # Recompute mask after fixing endpoints
    hole_mask = (signal == 0) | np.isnan(signal)

    # Interpolate interior holes using nearest valid neighbours
    if np.any(hole_mask):
        valid_idx = np.where(~hole_mask)[0]
        if len(valid_idx) > 0:
            signal[hole_mask] = np.interp(np.where(hole_mask)[0], valid_idx, signal[valid_idx])

    return signal[:end]


def lead_extraction(
    dic_tracks: dict[int, np.ndarray], extraction_method: str, TYPE: str, NOISE: bool | float, DEBUG: bool = False
) -> dict[str, np.ndarray]:
    """
    Extract the digital information from images

    Parameters
    ----------
    dic_tracks: dictionary, dictionary of track images
    extraction_method: str, it can be : "lazy", "full", "fragmented"
    TYPE  : str, format of the image
    NOISE : bool, if the image is noised or not
    DEBUG : bool, show the image

    Returns
    -------
    dictionary: dictionary of digital tracks
    """

    # Digital tracks dictionary
    dic_extracted_tracks = {}
    dic_image_bin = {}
    dic_extracted_track_not_scale = {}
    for d in dic_tracks:
        # Kardia Files are already binarize
        image_bin = dic_tracks[d]
        # Plot the binarized image
        if DEBUG:
            plt.imshow(image_bin)
            plt.show()
            plt.imshow(image_bin)

        # List of the extracted signal
        extraction = []

        if extraction_method == "lazy":
            extraction = lazy_extraction(image_bin)
        elif extraction_method == "full":
            extraction = full_extraction(image_bin)
        elif extraction_method == "fragmented":
            extraction = fragmented_extraction(image_bin)
        # Removing the holes in the signal
        signal = sup_holes(extraction, TYPE)

        # Scale the signal in time each tracks length 10sec with a frequency of 500hz it is 5000pts
        # by tracks plus the reference pulse
        x = [i for i in range(len(signal))]
        y = signal
        if TYPE.lower() == "classic":
            new_x = [i for i in np.arange(0, len(signal), len(signal) / SIGNAL_LENGTH_CLASSIC)]
        elif TYPE.lower() == "kardia":
            new_x = [i for i in np.arange(0, len(signal), len(signal) / SIGNAL_LENGTH_KARDIA)]
        else:
            new_x = [i for i in np.arange(0, len(signal), len(signal) / SIGNAL_LENGTH_STANDARD)]
        signal_scale = np.interp(new_x, x, y)
        dic_extracted_track_not_scale[d] = signal
        dic_extracted_tracks[d] = signal_scale
        dic_image_bin[d] = image_bin

        # Plot the signal before its scale
        if DEBUG:
            plt.plot(signal, c="r")
            plt.show()

    return (dic_extracted_tracks, dic_image_bin, dic_extracted_track_not_scale)


def lead_cutting(
    dic_tracks: dict[int, np.ndarray],
    DPI: int,
    TYPE: str,
    FORMAT: str,
    page: int,
    NOISE: bool | float,
    DEBUG: bool,
    dic_image_bin: dict[int, np.ndarray] | None = None,
) -> dict[str, np.ndarray] | np.ndarray:
    """
    Cut each tracks into leads

    Parameters
    ----------
    dic_tracks: dictionary, dictionary of track images
    DPI   : int, resolution
    TYPE  : str, format of the image
    NOISE : bool, if the image is noised or not
    DEBUG : bool, show the image
    dic_image_bin : dict, optional binary track images for calibration

    Returns
    -------
    dictionary: dictionary of leads
    """
    # Dictionary with reference pulse for each tracks
    dic_ref_pulse = {}
    # Dictionary with the lead
    dic_leads = {}
    LEAD_LENGTH = 0
    LEAD_NUMBER = 1
    dic_association = {0: "II"}
    # If the it is a classical format
    if TYPE.lower() == "classic" or (TYPE.lower() == "kardia" and FORMAT == "multilead"):
        if TYPE.lower() != "classic":
            if page == 0:
                # the reference pulse lasts 0.28sec
                LENGTH_PULSE = REF_PULSE_KARDIA
            else:
                LENGTH_PULSE = 0

        dic_time = {}
        # The disposition of the ECG is 4x4
        if len(dic_tracks) == 4:
            # leads lasts 2.5sec if there are 4 tracks
            LEAD_NUMBER = 4
            dic_association = {
                0: ["I", "AVR", "V1", "V4"],
                1: ["II", "AVL", "V2", "V5"],
                2: ["III", "AVF", "V3", "V6"],
                3: ["II"],
            }
            dic_time = LEAD_TIME_3X4
        # The disposition of the ECG is 6x2
        elif len(dic_tracks) == 6:
            # leads last 5sec if there are 6 tracks
            LEAD_NUMBER = 2
            dic_association = {
                0: ["I", "V1"],
                1: ["II", "V2"],
                2: ["III", "V3"],
                3: ["AVR", "V4"],
                4: ["AVL", "V5"],
                5: ["AVF", "V6"],
            }
            dic_time = LEAD_TIME_6X2

        ########## METTRE UN ELSE ICI ##################
        # else:
        ########## METTRE UN ELSE ICI ##################

        if TYPE.lower() == "kardia" and FORMAT.lower() == "multilead":
            # leads last 10sec in kardia

            dic_association = {
                0: "I",
                1: "II",
                2: "III",
                3: "AVR",
                4: "AVL",
                5: "AVF",
            }

        # Pre-compute calibration factors for all tracks.
        # When binary track images are available (dic_image_bin), measure the
        # calibration square height directly from the image for higher accuracy.
        # Then cross-validate across tracks to replace outlier values.
        _calib = {}  # {track_idx: (pixel_zero, f, LENGTH_PULSE)}
        if TYPE.lower() != "kardia":
            for t in dic_tracks:
                _lp = REF_PULSE_CLASSIC
                if len(dic_tracks) in (4, 6):
                    _lp = len(dic_tracks[t]) - SIGNAL_LENGTH_STANDARD

                # Prefer binary-image calibration (measures actual square height)
                if dic_image_bin is not None and t in dic_image_bin:
                    _pz, _f = _calibrate_from_binary(dic_image_bin[t], DPI)
                else:
                    _ref = dic_tracks[t][:_lp]
                    _pz = float(max(_ref))
                    _p1 = float(min(_ref))
                    _f = _pz - _p1
                _calib[t] = (_pz, max(_f, 0.001), _lp)
            # Cross-validate: replace outlier f values using median-based detection.
            # A track whose f deviates more than 2× from the median is replaced.
            if len(_calib) > 1:
                all_f = [v[1] for v in _calib.values()]
                median_f = float(np.median(all_f))
                for t in _calib:
                    pz, f_val, lp = _calib[t]
                    if f_val < 0.5 * median_f or f_val > 2.0 * median_f:
                        logger.info("Track %d: calibration f=%.1f replaced by median %.1f", t, f_val, median_f)
                        _calib[t] = (pz, median_f, lp)

        # Plot each tracks
        for t in dic_tracks:
            if DEBUG:
                LENGTH_PULSE = 140
                logger.debug("Track: %s", t)
                plt.figure(figsize=(20, 14))
                plt.plot(dic_tracks[t])
                plt.axvline(LENGTH_PULSE, c="r")

            if TYPE.lower() != "kardia":
                pixel_zero, f, LENGTH_PULSE = _calib[t]
                dic_ref_pulse[t] = dic_tracks[t][:LENGTH_PULSE]

                # Define the beggining of lead part
                LEAD_LENGTH = int(len(dic_tracks[t][LENGTH_PULSE:]) / LEAD_NUMBER)
                length = LENGTH_PULSE
                # length = LENGTH_PULSE
                # Define the lead position
                it = 0

                # special case on the disposition 4x4 the last track containe 10sec of the lead II
                if len(dic_tracks) == 4 and t == 3:
                    dic_leads["IIc"] = (
                        (pixel_zero - dic_tracks[t][LENGTH_PULSE : 4 * LEAD_LENGTH + LENGTH_PULSE]) / f
                    ) * AMPLITUDE_SCALE_UV
                    if DEBUG:
                        plt.show()

                # extract each lead from the tracks
                elif LEAD_LENGTH != 0:
                    while length < len(dic_tracks[t]):
                        try:
                            dic_leads[dic_association[t][it]] = (
                                (pixel_zero - dic_tracks[t][length : length + LEAD_LENGTH]) / f
                            ) * AMPLITUDE_SCALE_UV  # We fill the leads dictionnary with the name of the lead and the image of it
                            length += int(len(dic_tracks[t][LENGTH_PULSE:]) / LEAD_NUMBER)
                            it += 1
                            if DEBUG:
                                plt.axvline(length, c="r")
                        except Exception:
                            length += int(len(dic_tracks[t][LENGTH_PULSE:]) / LEAD_NUMBER)
                else:
                    return 0
                if DEBUG:
                    plt.show()

            else:
                if page == 0:
                    ref_pulse = dic_tracks[t][:LENGTH_PULSE]
                    # Pixel of amplitude 0mV
                    pixel_zero = max(ref_pulse)
                    # Pixel of amplitude 1mV
                    pixel_one = min(ref_pulse)
                    # Define the factor
                    f = pixel_zero - pixel_one
                    if f == 0:
                        f = 1
                    # Define the beggining of lead part
                    length = LENGTH_PULSE
                    dic_leads["ref"] = [pixel_zero, f]

                    # Scale the signal in amplitude
                    dic_leads[dic_association[t]] = ((pixel_zero - dic_tracks[t][length:]) / f) * AMPLITUDE_SCALE_UV

                else:
                    length = 0
                    dic_leads[dic_association[t]] = dic_tracks[t][length:]

        try:
            for k in dic_leads:
                zero_vector = np.zeros(SIGNAL_LENGTH_STANDARD)
                lead_data = dic_leads[k]
                t_start, t_end = dic_time[k]
                expected_len = t_end - t_start
                if len(lead_data) > expected_len:
                    lead_data = lead_data[:expected_len]
                elif len(lead_data) < expected_len:
                    padded = np.zeros(expected_len)
                    padded[: len(lead_data)] = lead_data
                    lead_data = padded
                zero_vector[t_start:t_end] = lead_data
                dic_leads[k] = zero_vector
        except Exception as e:
            logger.warning("Lead placement failed: %s", e)
        return dic_leads

    # If the format is not classic
    else:
        if TYPE.lower() == "apple":
            LENGTH_PULSE = REF_PULSE_APPLE
        elif TYPE.lower() == "kardia":
            LENGTH_PULSE = REF_PULSE_KARDIA
        else:
            LENGTH_PULSE = REF_PULSE_GENERIC

        for t in dic_tracks:
            if t == 0:
                # Plot each tracks
                if DEBUG:
                    plt.figure(figsize=(20, 14))
                    plt.plot(dic_tracks[t])
                    plt.axvline(LENGTH_PULSE, c="r")
                    plt.show()

                # Isolate and calibrate the reference pulse
                dic_ref_pulse = dic_tracks[t][:LENGTH_PULSE]
                pixel_zero, f = _calibrate_ref_pulse(dic_ref_pulse, DPI)

                # Separate the signal from the reference pulse
                all_signal = dic_tracks[t][LENGTH_PULSE:]

            # Concatane the signal if it is on more than one track
            else:
                dist = np.mean(all_signal) - np.mean(dic_tracks[t])
                all_signal = np.concatenate((all_signal, dic_tracks[t] + dist), axis=0)

            # Plot the different pixel
            if DEBUG:
                logger.debug("0: %s", pixel_zero)
                logger.debug("1: %s", pixel_one)
                logger.debug("1st pixel: %s", all_signal[0])

        # Scale the signal in amplitude
        new_signal = ((pixel_zero - all_signal) / f) * AMPLITUDE_SCALE_UV

        return new_signal
