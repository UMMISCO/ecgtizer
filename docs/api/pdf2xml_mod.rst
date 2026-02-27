ecgtizer.PDF2XML_mod
====================

.. automodule:: ecgtizer.PDF2XML_mod
   :members:
   :undoc-members:
   :show-inheritance:

Functional Examples
-------------------

Plot digitized leads
^^^^^^^^^^^^^^^^^^^^

:func:`plot_function` displays single-lead or 12-lead panel plots:

.. code-block:: python

   import numpy as np
   from ecgtizer.PDF2XML_mod import plot_function

   # Build a sample lead dictionary (12 leads, 5000 samples each)
   rng = np.random.RandomState(0)
   t = np.linspace(0, 10, 5000)
   lead_all = {}
   for name in ["I", "II", "III", "AVR", "AVL", "AVF",
                 "V1", "V2", "V3", "V4", "V5", "V6"]:
       lead_all[name] = (300 * np.sin(2 * np.pi * 1.2 * t + rng.rand())
                         + rng.randn(5000) * 20).astype(int)

   # Plot all 12 leads in a grid
   plot_function(lead_all)

   # Plot a single lead
   plot_function(lead_all, lead="II", c="blue")

   # Plot a time window and save to file
   plot_function(lead_all, lead="V1", b=0, e=2500, save="V1_segment.png")

Using plot_function with real ECGtizer output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from ecgtizer import ECGtizer
   from ecgtizer.PDF2XML_mod import plot_function

   ecg = ECGtizer("data/PTB-XL/PDF/00009_hr.pdf", dpi=500)
   plot_function(ecg.extracted_lead, lead="II", save="real_lead_II.png")

Overlay extracted waveforms on the original image
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from ecgtizer.PDF2XML_mod import plot_overlay

   # plot_overlay is called internally by ECGtizer.plot_over(),
   # but you can call it directly with the pipeline outputs:
   # plot_overlay(lead=dic_tracks_extracted, image=original_image,
   #              piqueh=variance_h, piquev=variance_v)

Write an HL7 aECG XML file
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   from ecgtizer.PDF2XML_mod import write_xml

   # Create a synthetic 12-lead ECG
   ecg_dict = {}
   t = np.linspace(0, 10, 5000)
   for name in ["I", "II", "III", "AVR", "AVL", "AVF",
                 "V1", "V2", "V3", "V4", "V5", "V6"]:
       ecg_dict[name] = (300 * np.sin(2 * np.pi * 1.2 * t)).astype(int)

   # Patient metadata table
   table = {
       "BPM": "75",
       "low_freq": "0.05",
       "high_freq": "150",
       "Inter PR (ms)": "160",
       "Dur.QRS (ms)": "90",
       "QT (ms)": "380",
       "QTc (ms)": "410",
       "Axe P": "60",
       "Axe R": "30",
       "Axe T": "45",
       "Moy RR (ms)": "800",
       "QTcB (ms)": "405",
       "QTcF (ms)": "400",
       "Rythme": "Sinus",
       "ECG": "Normal",
       "Age": "60",
       "sex": "M",
       "other_information": "Synthetic test ECG",
   }

   write_xml(ecg_dict, "output/synthetic.xml", TYPE="classic", table=table)

Write XML from an actual extraction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from ecgtizer import ECGtizer
   from ecgtizer.PDF2XML_mod import write_xml

   ecg = ECGtizer("data/PTB-XL/PDF/00075_hr.pdf", dpi=500)

   write_xml(
       matrix=ecg.extracted_lead,
       path_out="output/00075_hr.xml",
       TYPE=ecg.TYPE,
       table=ecg.table_parameters,
       num_version="1.0",
       date_version="27.02.2026",
   )

Convert a NumPy array to a space-separated string
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from ecgtizer.PDF2XML_mod import transform_np2txt

   import numpy as np
   arr = np.array([100, -50, 0, 230, 450])
   txt = transform_np2txt(arr)
   print(txt)   # "100 -50 0 230 450"

Parse a date/time string
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from ecgtizer.PDF2XML_mod import conversion_time

   date_str, time_str = conversion_time(
       day="15", month="Feb", year="2026", hour="14:30"
   )
   print(date_str)   # "20260215"
   print(time_str)   # "143000"
