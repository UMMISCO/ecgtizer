ecgtizer.PDF2XML
================

.. automodule:: ecgtizer.PDF2XML
   :members:
   :undoc-members:
   :show-inheritance:

Functional Examples
-------------------

This module contains the core image-processing pipeline that
:class:`~ecgtizer.ecgtizer.ECGtizer` orchestrates internally. Each
function can also be called directly for custom workflows.

Convert a PDF to images
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from ecgtizer.PDF2XML import convert_PDF2image

   pages, num_pages, success = convert_PDF2image(
       "data/PTB-XL/PDF/00009_hr.pdf", DPI=500
   )

   print(success)            # True
   print(num_pages)          # 1
   print(type(pages[0]))     # <class 'numpy.ndarray'>
   print(pages[0].shape)     # (h, w, 3) — RGB image

Detect noise type and ECG format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from ecgtizer.PDF2XML import check_noise_type

   image = pages[0]   # from convert_PDF2image
   TYPE, NOISE = check_noise_type(image, DPI=500)

   print(TYPE)    # "classic" | "kardia" | "wellue" | "apple"
   print(NOISE)   # True / False — noisy background detected

Extract text regions
^^^^^^^^^^^^^^^^^^^^

Mask patient-identifying text and annotations in the header area:

.. code-block:: python

   from ecgtizer.PDF2XML import text_extraction

   masked_image = text_extraction(image, TYPE, DPI=500, NOISE=NOISE)
   # masked_image has text regions replaced with white pixels

Segment into individual ECG tracks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from ecgtizer.PDF2XML import tracks_extraction

   dic_tracks, variance_h, variance_v = tracks_extraction(
       masked_image, TYPE, dpi=500, NOISE=NOISE
   )

   # dic_tracks: dict mapping track names → sub-images
   print(list(dic_tracks.keys()))
   # e.g. ['0', '1', '2', '3'] for a 3x4 classic layout

Binarize and extract waveforms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from ecgtizer.PDF2XML import lead_extraction

   dic_extracted, image_bin, not_scaled = lead_extraction(
       dic_tracks, "fragmented", TYPE, NOISE=NOISE
   )

   # dic_extracted: dict of track → extracted signal array
   for track_name, signal in dic_extracted.items():
       print(f"Track {track_name}: {len(signal)} samples")

Calibrate and assign lead names
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from ecgtizer.PDF2XML import lead_cutting

   dic_lead = lead_cutting(
       dic_extracted, dpi=500, TYPE=TYPE, page=0, NOISE=NOISE
   )

   # dic_lead: dict mapping standard lead names → calibrated signals
   print(list(dic_lead.keys()))
   # ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', ...]

Fill missing signal points
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   from ecgtizer.PDF2XML import sup_holes

   # Example: a signal with a "hole" (flat zero region)
   signal = [100, 120, 0, 0, 0, 130, 150]
   filled = sup_holes(signal)
   print(filled)
   # The zero region is linearly interpolated

Full manual pipeline
^^^^^^^^^^^^^^^^^^^^

Run the complete pipeline step by step:

.. code-block:: python

   from ecgtizer.PDF2XML import (
       convert_PDF2image, check_noise_type, text_extraction,
       tracks_extraction, lead_extraction, lead_cutting, clean_tracks,
   )

   # 1. PDF to image
   pages, _, success = convert_PDF2image(
       "data/PTB-XL/PDF/00075_hr.pdf", DPI=500
   )
   image = pages[0]

   # 2. Detect format and noise
   TYPE, NOISE = check_noise_type(image, DPI=500)

   # 3. Mask text
   masked = text_extraction(image, TYPE, DPI=500, NOISE=NOISE)

   # 4. Segment tracks
   dic_tracks, vh, vv = tracks_extraction(masked, TYPE, dpi=500, NOISE=NOISE)

   # 5. Clean tracks (remove small noise blobs)
   dic_tracks = clean_tracks(dic_tracks, TYPE, NOISE=NOISE)

   # 6. Extract waveforms
   dic_extracted, bins, not_scaled = lead_extraction(
       dic_tracks, "fragmented", TYPE, NOISE=NOISE
   )

   # 7. Calibrate + name
   dic_lead = lead_cutting(dic_extracted, dpi=500, TYPE=TYPE, page=0, NOISE=NOISE)

   # Result
   for name, sig in dic_lead.items():
       print(f"{name}: {len(sig)} samples, range [{min(sig):.0f}, {max(sig):.0f}]")
