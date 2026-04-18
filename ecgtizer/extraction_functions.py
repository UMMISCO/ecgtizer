"""Waveform extraction algorithms for binarized ECG track images.

Three strategies with different speed/accuracy trade-offs:

* **lazy** -- fast, noise-tolerant, but smooths peaks.
* **full** -- fast, high fidelity, but may include annotation artifacts.
* **fragmented** -- slower, highest fidelity via contour detection.
"""

from __future__ import annotations

import numpy as np


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
    # Find anchor point: average of all lit pixels in the first column
    first_col_lit = np.where(image_bin[:, 0] == 255)[0]
    if len(first_col_lit) > 0:
        anchor = int(np.mean(first_col_lit))
    else:
        anchor = image_bin.shape[0] // 2
    signal = [anchor]

    # We then go through the image column by column, looking for the lit pixel closest to the anchor pixel.
    for i in range(1, len(image_bin[0])):
        # If we can stay at the same level as the anchor pixel, we do so
        if image_bin[anchor, i] == 255:
            signal.append(anchor)
        else:
            # Otherwise we look for the nearest lit pixel at the top and bottom, and as soon as we find one we stop and store it.
            # We search within a window of 1000 pixels to avoid searching too far.
            try:
                for j in range(1000):
                    if image_bin[anchor + j, i] == 255:
                        signal.append(anchor + j)
                        anchor = anchor + j
                        break
                    elif image_bin[anchor - j, i] == 255:
                        signal.append(anchor - j)
                        anchor = anchor - j
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
    # Vectorized: mean row position of lit pixels per column
    mask = image_bin == 255
    row_indices = np.arange(image_bin.shape[0], dtype=float)
    # Weighted sum of row indices where mask is True, per column
    weighted_sum = np.dot(row_indices, mask)  # shape: (width,)
    count = mask.sum(axis=0).astype(float)  # shape: (width,)
    # Avoid division by zero: where no lit pixels, return 0.0
    extraction = np.divide(weighted_sum, count, out=np.zeros(image_bin.shape[1]), where=count > 0)
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
    # Extract the signal fragment per column using vectorized grouping.
    h, w = image_bin.shape
    midpoint = h / 2.0
    signal = np.full(w, midpoint)

    for i in range(w):
        positions = np.where(image_bin[:, i] == 255)[0]
        if len(positions) == 0:
            signal[i] = signal[i - 1] if i > 0 else midpoint
            continue
        if len(positions) == 1:
            signal[i] = float(positions[0])
            continue
        # Split into fragments at gaps (consecutive diff > 1)
        breaks = np.where(np.diff(positions) > 1)[0] + 1
        if len(breaks) == 0:
            # Single contiguous fragment
            signal[i] = np.mean(positions)
        else:
            # Last fragment is the signal (first fragments are text labels)
            signal[i] = np.mean(positions[breaks[-1] :])
    return signal.tolist()
