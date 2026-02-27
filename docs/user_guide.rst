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

.. code-block:: bash

   python ECGtizer_main.py "ecg.pdf" 500 "fragmented" --verbose "output.xml"
