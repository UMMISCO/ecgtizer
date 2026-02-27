ecgtizer.ecgtizer
=================

.. automodule:: ecgtizer.ecgtizer
   :members:
   :undoc-members:
   :show-inheritance:

Functional Examples
-------------------

Digitize a 12-lead ECG PDF
^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`ECGtizer` class is the main entry point. Pass a PDF path,
resolution, and extraction method to run the full pipeline:

.. code-block:: python

   from ecgtizer import ECGtizer

   ecg = ECGtizer(
       file="data/PTB-XL/PDF/00009_hr.pdf",
       dpi=500,
       extraction_method="fragmented",
       verbose=True,
   )

   # Check that the conversion succeeded
   print(ecg.good)   # True
   print(ecg.TYPE)    # "classic"

   # Digitized leads are stored as a dict of NumPy arrays
   leads = ecg.extracted_lead
   print(list(leads.keys()))
   # ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

   print(leads["II"].shape)   # (5140,)  — ~10.3 s at 500 Hz

Compare extraction methods
^^^^^^^^^^^^^^^^^^^^^^^^^^

Each method trades speed for accuracy:

.. code-block:: python

   pdf = "data/PTB-XL/PDF/00075_hr.pdf"

   ecg_lazy = ECGtizer(pdf, dpi=500, extraction_method="lazy")
   ecg_full = ECGtizer(pdf, dpi=500, extraction_method="full")
   ecg_frag = ECGtizer(pdf, dpi=500, extraction_method="fragmented")

   for name, obj in [("lazy", ecg_lazy), ("full", ecg_full), ("frag", ecg_frag)]:
       print(f"{name}: lead II length = {len(obj.extracted_lead['II'])}")

Plot single and multi-lead panels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Plot all 12 leads in a grid
   ecg.plot()

   # Plot a single lead with a sample range
   ecg.plot(lead="V1", begin=0, end=2500, c="blue")

   # Save to file
   ecg.plot(lead="II", save="lead_II.png", transparent=True)

   # Plot completed leads (after running completion)
   ecg.plot(completion=True)

Overlay digitized traces on the original image
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Superpose extracted waveforms on the source ECG image
   ecg.plot_over()

Export to HL7 aECG XML
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   ecg.save_xml(
       save="output/00009_hr.xml",
       num_version="1.0",
       date_version="27.02.2026",
   )

Deep-learning lead completion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Extend partial leads (2.5 s or 5 s) to the full 10-second duration:

.. code-block:: python

   import torch

   device = "cuda" if torch.cuda.is_available() else "cpu"
   ecg.completion(path_model="model/Model_Completion.pth", device=device)

   # The completed signals are now in ecg.extracted_lead_completed
   print(list(ecg.extracted_lead_completed.keys()))

   # Save the completed ECG
   ecg.save_xml("output/00009_hr_completed.xml")

Use a 6x2 layout PDF
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   ecg_6x2 = ECGtizer(
       file="data/PTB-XL/PDF/00009_hr_6x2.pdf",
       dpi=500,
       extraction_method="full",
   )
   print(ecg_6x2.TYPE)   # "classic"
   print(list(ecg_6x2.extracted_lead.keys()))

Progress callback
^^^^^^^^^^^^^^^^^

Track pipeline progress for integration in a GUI:

.. code-block:: python

   def on_progress(message):
       print(f"[ECGtizer] {message}")

   ecg = ECGtizer(
       file="data/PTB-XL/PDF/00075_hr.pdf",
       dpi=500,
       extraction_method="full",
       Callback=on_progress,
   )
   # Prints: [ECGtizer] Converting PDF to image...
   #         [ECGtizer] Detecting noise and format...
   #         ...
