ecgtizer.anonymisation
======================

.. automodule:: ecgtizer.anonymisation
   :members:
   :undoc-members:
   :show-inheritance:

Functional Examples
-------------------

Anonymize an ECG PDF
^^^^^^^^^^^^^^^^^^^^

Remove patient-identifying text (name, date of birth, ID) from the
header area of an ECG PDF:

.. code-block:: python

   from ecgtizer.anonymisation import anonymisation

   anonymisation(
       file="data/PTB-XL/PDF/00009_hr.pdf",
       out="output/00009_hr_anon.pdf",
   )

The function:

1. Converts the PDF to an image at 300 DPI
2. Detects text-like dark regions in the upper-left corner
3. Applies morphological dilation to expand the detected regions
4. Masks them with white pixels
5. Exports the result as a new PDF

Batch anonymization
^^^^^^^^^^^^^^^^^^^

Process multiple files in a directory:

.. code-block:: python

   import os
   from ecgtizer.anonymisation import anonymisation

   input_dir  = "data/PTB-XL/PDF"
   output_dir = "output/anonymized"
   os.makedirs(output_dir, exist_ok=True)

   for filename in os.listdir(input_dir):
       if filename.endswith(".pdf"):
           anonymisation(
               file=os.path.join(input_dir, filename),
               out=os.path.join(output_dir, filename),
           )
           print(f"Anonymized: {filename}")

Convert a NumPy array to PDF
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`array_to_pdf` is a utility for saving any image array as a PDF:

.. code-block:: python

   import numpy as np
   from ecgtizer.anonymisation import array_to_pdf

   # Create a white image with a black rectangle
   img = np.ones((600, 800, 3), dtype=np.uint8) * 255
   img[100:200, 100:300] = [0, 0, 0]

   array_to_pdf(img, "output/test_image.pdf")

   # Also works with grayscale arrays
   gray = np.ones((400, 600), dtype=np.uint8) * 200
   array_to_pdf(gray, "output/gray_image.pdf")

Anonymization + digitization pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A common workflow: anonymize first, then digitize:

.. code-block:: python

   from ecgtizer.anonymisation import anonymisation
   from ecgtizer import ECGtizer

   # Step 1: remove patient info
   anonymisation("input_ecg.pdf", "anon_ecg.pdf")

   # Step 2: digitize the anonymized PDF
   ecg = ECGtizer("anon_ecg.pdf", dpi=500, extraction_method="fragmented")

   # Step 3: export as XML
   if ecg.good:
       ecg.save_xml("digitized.xml")
