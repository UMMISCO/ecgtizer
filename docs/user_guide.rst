User Guide
==========

Extraction Methods
------------------

ECGtizer provides three waveform extraction algorithms:

.. list-table::
   :header-rows: 1
   :widths: 15 15 15 55

   * - Method
     - Speed
     - Accuracy
     - Description
   * - ``lazy``
     - Fast
     - Moderate
     - Follows the nearest lit pixel from an anchor point. Good noise
       tolerance but smooths peaks.
   * - ``full``
     - Fast
     - High
     - Averages all lit-pixel positions per column. Captures more detail
       but may include annotation artifacts.
   * - ``fragmented``
     - Slower
     - Highest
     - Uses contour detection to separate signal from text labels.
       Best fidelity for clean recordings.

Choose the method via the ``extraction_method`` parameter:

.. code-block:: python

   ecg = ECGtizer("ecg.pdf", dpi=500, extraction_method="fragmented")

Lead Completion
---------------

Partial leads (2.5 s or 5 s) can be extended to the full 10-second
duration using a pre-trained convolutional autoencoder:

.. code-block:: python

   import torch

   device = "cuda" if torch.cuda.is_available() else "cpu"
   ecg.completion(path_model="model/Model_Completion.pth", device=device)

   # Plot the completed leads
   ecg.plot(completion=True)

   # Save completed leads to XML
   ecg.save_xml("completed.xml")

Signal Analysis
---------------

Compare a digitized ECG against the original recording:

.. code-block:: python

   from ecgtizer import analyse, BlandAltman, scatter_plot, overlap_plot

   # Correlation, RMSE, DTW metrics per lead
   results = analyse("digitized.xml", "original.xml")

   # Bland-Altman agreement plots
   BlandAltman("digitized.xml", "original.xml")

   # Scatter plots with linear regression
   scatter_plot("digitized.xml", "original.xml")

   # Overlay plots
   overlap_plot("digitized.xml", "original.xml", lead="II")

XML to PDF
----------

Re-render an HL7 aECG XML file as a publication-quality PDF:

.. code-block:: python

   from ecgtizer import xml_to_pdf

   xml_to_pdf("digitized.xml", "output.pdf", type_of_pdf="type1")

Supported layouts: ``"type1"`` (3x4) or ``"type2"`` (6x2).

PDF Anonymization
-----------------

Remove patient-identifying text from an ECG PDF:

.. code-block:: python

   from ecgtizer import anonymisation

   anonymisation("original.pdf", "anonymized.pdf")

Command Line
------------

After installation (``pip install ecgtizer``), the ``ecgtizer`` command is
available:

.. code-block:: bash

   ecgtizer ecg.pdf 500 fragmented output.xml --verbose

   # Force a specific ECG format
   ecgtizer ecg.png 300 full output.xml --type kardia

   # Show help
   ecgtizer --help


Real-World Workflows
--------------------

The following examples use the sample data included in the repository
(``data/PTB-XL/``).

End-to-end: PDF to XML to PDF
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Digitize a 12-lead ECG from PDF, save as XML, and re-render as a new PDF:

.. code-block:: python

   from ecgtizer import ECGtizer, xml_to_pdf

   # 1. Digitize
   ecg = ECGtizer(
       file="data/PTB-XL/PDF/00009_hr.pdf",
       dpi=500,
       extraction_method="fragmented",
       verbose=True,
   )

   # 2. Inspect
   print(f"Format: {ecg.TYPE}")
   print(f"Leads: {list(ecg.extracted_lead.keys())}")
   for name, sig in ecg.extracted_lead.items():
       print(f"  {name}: {len(sig)} samples")

   # 3. Export to XML
   ecg.save_xml("output/00009_hr.xml")

   # 4. Re-render as PDF (3x4 layout)
   xml_to_pdf("output/00009_hr.xml", "output/00009_hr_redrawn.pdf", "type1")

Compare digitized vs original recording
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Evaluate extraction accuracy using the PTB-XL ground truth:

.. code-block:: python

   from ecgtizer import ECGtizer, analyse, BlandAltman, scatter_plot

   # Digitize
   ecg = ECGtizer("data/PTB-XL/PDF/00075_hr.pdf", dpi=500,
                   extraction_method="fragmented")
   ecg.save_xml("output/00075_hr.xml")

   # Compare against original CSV
   results = analyse(
       "data/PTB-XL/Original/00075_hr.csv",
       "output/00075_hr.xml",
   )

   print("Lead | Correlation | MSE")
   print("---- | ----------- | ---")
   for lead, m in results.items():
       print(f"{lead:4s} | {m['correlation']:11.3f} | {m['MSE']:.1f}")

   # Visualize
   BlandAltman("data/PTB-XL/Original/00075_hr.csv", "output/00075_hr.xml",
               lead="II", save="output/bland_altman_II.png")
   scatter_plot("data/PTB-XL/Original/00075_hr.csv", "output/00075_hr.xml",
                lead="II", save="output/scatter_II.png")

Completion + analysis pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Digitize, complete partial leads, then compare with the original:

.. code-block:: python

   import torch
   from ecgtizer import ECGtizer, analyse

   ecg = ECGtizer("data/PTB-XL/PDF/00075_hr.pdf", dpi=500)

   device = "cuda" if torch.cuda.is_available() else "cpu"
   ecg.completion(path_model="model/Model_Completion.pth", device=device)

   ecg.save_xml("output/00075_hr_completed.xml")

   results = analyse(
       "data/PTB-XL/Original/00075_hr.csv",
       "output/00075_hr_completed.xml",
   )
   for lead, m in results.items():
       print(f"{lead}: correlation={m['correlation']:.3f}")

CSV to PDF rendering
^^^^^^^^^^^^^^^^^^^^

Generate publication-quality ECG PDFs from CSV signal files:

.. code-block:: python

   import pandas as pd
   from ecgtizer.XML2PDF import Write_PDF

   df = pd.read_csv("data/PTB-XL/Original/00129_hr.csv")

   lead_map = {
       "I": "I", "II": "II", "III": "III",
       "aVR": "AVR", "aVL": "AVL", "aVF": "AVF",
       "V1": "V1", "V2": "V2", "V3": "V3",
       "V4": "V4", "V5": "V5", "V6": "V6",
   }

   ecg = {}
   for csv_col, std_name in lead_map.items():
       ecg[std_name] = df[csv_col].values * 1000  # mV to uV

   # 6x2 layout
   Write_PDF(ecg, "output/00129_type2.pdf", type_of_pdf="type2")

   # 3x4 layout with rhythm strip
   Write_PDF(ecg, "output/00129_type1.pdf", type_of_pdf="type1",
             lead_IIc=ecg["II"])

Batch processing
^^^^^^^^^^^^^^^^

Process all PDFs in a directory:

.. code-block:: python

   import os
   from ecgtizer import ECGtizer

   input_dir  = "data/PTB-XL/PDF"
   output_dir = "output/batch"
   os.makedirs(output_dir, exist_ok=True)

   for filename in os.listdir(input_dir):
       if not filename.endswith(".pdf"):
           continue
       path_in = os.path.join(input_dir, filename)
       path_out = os.path.join(output_dir, filename.replace(".pdf", ".xml"))

       ecg = ECGtizer(path_in, dpi=500, extraction_method="full")
       if ecg.good:
           ecg.save_xml(path_out)
           print(f"OK:   {filename} -> {list(ecg.extracted_lead.keys())}")
       else:
           print(f"FAIL: {filename}")
