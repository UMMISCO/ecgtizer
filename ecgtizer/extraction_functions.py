"""Waveform extraction algorithms for binarized ECG track images.

Three strategies with different speed/accuracy trade-offs:

* **lazy** -- fast, noise-tolerant, but smooths peaks.
* **full** -- fast, high fidelity, but may include annotation artifacts.
* **fragmented** -- slower, highest fidelity via contour detection.
"""
from __future__ import annotations

import numpy as np
import cv2

def lazy_extraction(image_bin: np.ndarray) -> list[int]:
    """Extract a waveform by following the nearest lit pixel from an anchor.

    Fast and noise-tolerant. Starts from the average lit-pixel position
    in the first column and walks column-by-column, always jumping to the
    closest lit pixel within a 1000-pixel window.

    Parameters
    ----------
    image_bin : numpy.ndarray
        Binarized track image (255 = signal, 0 = background).

    Returns
    -------
    list[int]
        Vertical pixel positions representing the extracted waveform.
    """
    # We define a starting pixel which corresponds to our anchor point - the extraction will start from this point
    # We look for all the lit pixels in the first column and average over them
    first_pixel_position = []
    for j in range(len(image_bin)):
        if image_bin[j,0] == 255:
            first_pixel_position.append(j)
    try:
        anchor = int(np.mean(first_pixel_position))
        signal = [anchor]
    except ValueError:
        anchor = int(len(image_bin)/2)
        signal = [anchor]

    # We then go through the image column by column, looking for the lit pixel closest to the anchor pixel.
    for i in range(1,len(image_bin[0])):
        # If we can stay at the same level as the anchor pixel, we do so
        if image_bin[anchor,i] == 255:
            signal.append(anchor)
        else:
            # Otherwise we look for the nearest lit pixel at the top and bottom, and as soon as we find one we stop and store it.
            # We search within a window of 1000 pixels to avoid searching too far.
            try:
                for j in range(1000):
                    if image_bin[anchor+j,i] == 255:
                        signal.append(anchor+j)
                        anchor = anchor+j
                        break
                    elif image_bin[anchor-j,i] == 255:
                        signal.append(anchor-j)
                        anchor = anchor-j
                        break
            except IndexError:
                signal.append(anchor)
    return signal


def full_extraction(image_bin: np.ndarray) -> np.ndarray:
    """Extract a waveform by averaging all lit-pixel positions per column.

    Fast with high fidelity. Computes the mean row position of all lit
    pixels in each column. May include annotation artifacts if text
    overlaps the signal region.

    Parameters
    ----------
    image_bin : numpy.ndarray
        Binarized track image (255 = signal, 0 = background).

    Returns
    -------
    numpy.ndarray
        Mean vertical positions per column.
    """
    # We look at all the columns in the image and average the position of the lit pixels
    extraction = np.array([sum(i for i, valeur in enumerate(ligne) if valeur == 255) / (ligne.count(255)+0.01) for ligne in image_bin.T.tolist()])
    return extraction



def fragmented_extraction(image_bin: np.ndarray) -> list[float]:
    """Extract a waveform using contour-based fragment detection.

    Slower but highest fidelity. Groups consecutive lit pixels into
    fragments per column. When multiple fragments exist (e.g. signal
    plus text label), selects the last fragment as the signal.

    Parameters
    ----------
    image_bin : numpy.ndarray
        Binarized track image (255 = signal, 0 = background).

    Returns
    -------
    list[float]
        Mean vertical positions per column (from the signal fragment).
    """
    # Look at all the columns in the image and store the lit pixels.
    # if there's a gap between two lit pixels, we store them in a new list
    midpoint = len(image_bin) / 2
    signal = []
    for i in range(len(image_bin[0])):
        positions = np.where(image_bin[:,i] == 255)[0]
        # No lit pixels in this column — use previous value or midpoint
        if len(positions) == 0:
            signal.append(signal[-1] if signal else midpoint)
            continue
        # Group consecutive lit pixels into fragments
        matrix = []
        sub_list = [positions[0]]
        for j in range(1, len(positions)):
            if positions[j] == positions[j - 1] + 1:
                sub_list.append(positions[j])
            else:
                matrix.append(sub_list)
                sub_list = [positions[j]]
        matrix.append(sub_list)
        # The last fragment is assumed to be the signal (first fragments are text labels)
        if len(matrix) > 1:
            signal.append(np.mean(matrix[-1]))
        else:
            signal.append(np.mean(matrix[0]))
    return signal



