ecgtizer.analyses
=================

.. automodule:: ecgtizer.analyses
   :members:
   :undoc-members:
   :show-inheritance:

Functional Examples
-------------------

Compare a digitized ECG against the original
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`analyse` aligns signals with DTW and computes correlation
and MSE metrics per lead:

.. code-block:: python

   from ecgtizer.analyses import analyse

   results = analyse(
       "data/PTB-XL/Original/00121_hr.csv",
       "data/PTB-XL/Digitized/00121_hr.xml",
   )

   # Results: dict[lead_name -> dict with 'correlation', 'MSE', 'DTW']
   for lead, metrics in results.items():
       print(f"{lead}: r={metrics['correlation']:.3f}, "
             f"MSE={metrics['MSE']:.1f}, DTW={metrics['DTW']:.1f}")

Bland-Altman agreement plot
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Visualize measurement agreement between two recordings:

.. code-block:: python

   from ecgtizer.analyses import BlandAltman

   # Plot all leads
   BlandAltman(
       "data/PTB-XL/Original/00121_hr.csv",
       "data/PTB-XL/Digitized/00121_hr.xml",
   )

   # Plot a single lead
   BlandAltman(
       "data/PTB-XL/Original/00121_hr.csv",
       "data/PTB-XL/Digitized/00121_hr.xml",
       lead="II",
   )

   # Save to file
   BlandAltman(
       "data/PTB-XL/Original/00121_hr.csv",
       "data/PTB-XL/Digitized/00121_hr.xml",
       lead="V1",
       save="bland_altman_V1.png",
   )

Scatter plot with linear regression
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from ecgtizer.analyses import scatter_plot

   # Single lead
   scatter_plot(
       "data/PTB-XL/Original/00121_hr.csv",
       "data/PTB-XL/Digitized/00121_hr.xml",
       lead="II",
       save="scatter_II.png",
   )

   # All leads
   scatter_plot(
       "data/PTB-XL/Original/00121_hr.csv",
       "data/PTB-XL/Digitized/00121_hr.xml",
   )

Overlap (overlay) plot
^^^^^^^^^^^^^^^^^^^^^^

Superimpose original and digitized signals on the same axes:

.. code-block:: python

   from ecgtizer.analyses import overlap_plot

   overlap_plot(
       "data/PTB-XL/Original/00121_hr.csv",
       "data/PTB-XL/Digitized/00121_hr.xml",
       lead="II",
   )

   # Save all leads
   overlap_plot(
       "data/PTB-XL/Original/00121_hr.csv",
       "data/PTB-XL/Digitized/00121_hr.xml",
       save="overlap_all.png",
   )

Read an HL7 aECG XML file
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from ecgtizer.analyses import read_xml

   leads = read_xml("data/PTB-XL/Digitized/00121_hr.xml")

   print(type(leads))          # <class 'dict'>
   print(list(leads.keys()))   # ['I', 'II', 'III', 'AVR', ...]
   print(leads["II"].shape)    # (5140,) -- NumPy int array

Parse a space-separated digit string
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from ecgtizer.analyses import read_lead

   values = read_lead("100 200 -50 abc 300")
   print(values)   # [100, 200, -50, 0, 300]  (invalid token -> 0)

Align two signals with DTW
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   from ecgtizer.analyses import alignement

   # Two signals with a slight time shift
   t = np.linspace(0, 2 * np.pi, 500)
   sig1 = np.sin(t)
   sig2 = np.sin(t + 0.3)  # shifted by ~0.3 radians

   aligned1, aligned2 = alignement(sig1, sig2)
   print(f"Aligned lengths: {len(aligned1)}, {len(aligned2)}")

Using dict inputs instead of file paths
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All analysis functions accept either file paths (CSV/XML) or
pre-loaded dictionaries:

.. code-block:: python

   import numpy as np
   from ecgtizer.analyses import analyse, BlandAltman, overlap_plot

   # Build two synthetic ECG dicts
   t = np.linspace(0, 10, 5000)
   original  = {name: np.sin(t * (i + 1)) * 300
                 for i, name in enumerate(["I", "II", "III", "AVR",
                 "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"])}
   digitized = {name: sig + np.random.randn(5000) * 10
                 for name, sig in original.items()}

   results = analyse(original, digitized)
   BlandAltman(original, digitized, lead="II")
   overlap_plot(original, digitized, lead="V1")
