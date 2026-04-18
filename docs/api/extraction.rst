ecgtizer.extraction_functions
=============================

.. automodule:: ecgtizer.extraction_functions
   :members:
   :undoc-members:
   :show-inheritance:

Functional Examples
-------------------

These three functions operate on **binarized track images** (NumPy arrays
where ``255`` = signal pixel, ``0`` = background). In normal usage,
:class:`~ecgtizer.ecgtizer.ECGtizer` calls them internally, but you can
use them directly for custom workflows.

Create a synthetic binary track
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np

   # Simulate a 50-pixel-tall track with a sine wave
   h, w = 50, 500
   image_bin = np.zeros((h, w), dtype=np.uint8)

   for x in range(w):
       y = int(h / 2 + 15 * np.sin(2 * np.pi * x / 100))
       y = max(0, min(h - 1, y))
       image_bin[y, x] = 255

Lazy extraction
^^^^^^^^^^^^^^^

Follows the nearest lit pixel from an anchor point. Fast and
noise-tolerant, but smooths sharp peaks:

.. code-block:: python

   from ecgtizer.extraction_functions import lazy_extraction

   signal = lazy_extraction(image_bin)

   print(type(signal))   # <class 'list'>
   print(len(signal))    # 500 — one value per column

Full extraction
^^^^^^^^^^^^^^^

Averages all lit-pixel positions per column. Higher fidelity but may
include annotation artifacts:

.. code-block:: python

   from ecgtizer.extraction_functions import full_extraction

   signal = full_extraction(image_bin)

   print(type(signal))    # <class 'numpy.ndarray'>
   print(signal.shape)    # (500,)

Fragmented extraction
^^^^^^^^^^^^^^^^^^^^^

Uses contour analysis to separate signal from text labels.
Best fidelity for clean, high-quality recordings:

.. code-block:: python

   from ecgtizer.extraction_functions import fragmented_extraction

   signal = fragmented_extraction(image_bin)

   print(type(signal))   # <class 'list'>
   print(len(signal))    # 500

Compare all three methods
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import matplotlib.pyplot as plt
   from ecgtizer.extraction_functions import (
       lazy_extraction, full_extraction, fragmented_extraction,
   )

   lazy  = lazy_extraction(image_bin)
   full  = full_extraction(image_bin)
   frag  = fragmented_extraction(image_bin)

   fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
   for ax, sig, name in zip(axes, [lazy, full, frag],
                             ["Lazy", "Full", "Fragmented"]):
       ax.plot(sig)
       ax.set_ylabel(name)
       ax.invert_yaxis()   # pixel coords: 0 = top
   plt.xlabel("Column (pixels)")
   plt.tight_layout()
   plt.show()

Using extraction on a real ECG track
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After running the pipeline steps manually:

.. code-block:: python

   from ecgtizer.PDF2XML import (
       convert_PDF2image, check_noise_type, text_extraction,
       tracks_extraction, lead_extraction,
   )

   # Step 1 — PDF to image
   pages, _, _ = convert_PDF2image("data/PTB-XL/PDF/00075_hr.pdf", DPI=500)
   image = pages[0]

   # Step 2 — detect format and noise
   TYPE, NOISE = check_noise_type(image, DPI=500)

   # Step 3 — segment into tracks
   dic_tracks, vh, vv = tracks_extraction(image, TYPE, dpi=500, NOISE=NOISE)

   # Step 4 — binarize and extract with each method
   dic_lazy, _, _ = lead_extraction(dic_tracks, "lazy", TYPE, NOISE=NOISE)
   dic_full, _, _ = lead_extraction(dic_tracks, "full", TYPE, NOISE=NOISE)
   dic_frag, _, _ = lead_extraction(dic_tracks, "fragmented", TYPE, NOISE=NOISE)
