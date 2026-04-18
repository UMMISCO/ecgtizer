Architecture
============

Processing Pipeline
-------------------

ECGtizer processes ECG documents through the following stages::

    PDF / Image
         |
         v
   convert_PDF2image     pdf2image + poppler
         |
         v
   check_noise_type      Variance analysis on image rows/columns
         |                Detects: clean / noisy / partial noise
         v
   text_extraction        Mask header text and annotations
         |
         v
   tracks_extraction      Horizontal/vertical variance peaks
         |                Splits image into individual ECG strips
         v
   lead_extraction        Binarize + extract waveform per strip
         |                Uses selected method (lazy/full/fragmented)
         v
   lead_cutting           Calibrate amplitude using reference pulse
         |                Segment strips into named leads
         v
   write_xml              HL7 aECG XML serialization

   Optional:
   completion_            PyTorch autoencoder extends partial leads
                          to full 10-second recordings

Module Organization
-------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Module
     - Responsibility
   * - ``ecgtizer.ecgtizer``
     - Main :class:`~ecgtizer.ecgtizer.ECGtizer` class orchestrating
       the pipeline.
   * - ``ecgtizer.PDF2XML``
     - Core image processing: noise detection, binarization, track
       segmentation, waveform extraction, amplitude calibration.
   * - ``ecgtizer.extraction_functions``
     - Three extraction algorithms: lazy, full, fragmented.
   * - ``ecgtizer.PDF2XML_mod``
     - Plotting helpers, XML serialization, signal utilities.
   * - ``ecgtizer.completion``
     - PyTorch autoencoder model and completion pipeline.
   * - ``ecgtizer.analyses``
     - Signal comparison: DTW, Pearson correlation, Bland-Altman,
       scatter plots, overlap plots.
   * - ``ecgtizer.XML2PDF``
     - XML-to-PDF rendering with ReportLab (graph paper, lead plots).
   * - ``ecgtizer.anonymisation``
     - PDF anonymization via morphological text detection and masking.

Supported ECG Formats
---------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 40

   * - Format
     - Layout
     - Leads
     - Source
   * - Classic 3x4
     - 4 rows, 3 cols
     - I, II, III, aVR, aVL, aVF, V1-V6
     - Standard 12-lead printout
   * - Classic 6x2
     - 2 rows, 6 cols
     - Same 12 leads
     - Alternative 12-lead layout
   * - Wellue
     - Single strip
     - I
     - Wellue portable devices
   * - Kardia single
     - Single strip
     - I
     - AliveCor Kardia single-lead
   * - Kardia multi
     - Multiple pages
     - I, II, III, aVR, aVL, aVF
     - AliveCor Kardia 6-lead
   * - Apple Watch
     - Single strip
     - I
     - Apple Watch ECG export

Completion Model
----------------

The lead completion module uses a dual-path convolutional autoencoder
that combines 1-D (per-lead) and 2-D (cross-lead) convolutions:

1. **Encoder**: Four stages of parallel 1-D and 2-D convolutions with
   batch normalization, LeakyReLU and dropout. Feature maps are
   concatenated at each level.

2. **Transition block**: A 2-D convolution that processes the bottleneck
   representation.

3. **Decoder**: Four stages of transposed convolutions with skip
   connections from the corresponding encoder levels.

Input shape: ``(batch, 1, 12, 512)`` — 12 leads, 512 time steps.
Output shape: ``(batch, 12, 512)`` — completed 12-lead signal.

The completed signal is denormalized and resampled back to 5000 samples
(10 seconds at 500 Hz).
