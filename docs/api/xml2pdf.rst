ecgtizer.XML2PDF
================

.. automodule:: ecgtizer.XML2PDF
   :members:
   :undoc-members:
   :show-inheritance:

Functional Examples
-------------------

Convert an XML file to a PDF
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The simplest entry point is :func:`xml_to_pdf`:

.. code-block:: python

   from ecgtizer.XML2PDF import xml_to_pdf

   # 3x4 layout (type1)
   xml_to_pdf(
       "data/PTB-XL/Digitized/00121_hr.xml",
       "output/00121_type1.pdf",
       type_of_pdf="type1",
   )

   # 6x2 layout (type2)
   xml_to_pdf(
       "data/PTB-XL/Digitized/00121_hr.xml",
       "output/00121_type2.pdf",
       type_of_pdf="type2",
   )

Convert a completed ECG to PDF
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from ecgtizer.XML2PDF import xml_to_pdf

   xml_to_pdf(
       "data/PTB-XL/Digitized/00121_hr_completed.xml",
       "output/00121_completed.pdf",
   )

Read leads from an XML file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from ecgtizer.XML2PDF import read_xml

   ecg = read_xml("data/PTB-XL/Digitized/00075_hr.xml")

   print(type(ecg))           # <class 'dict'>
   print(list(ecg.keys()))    # ['I', 'II', 'III', 'AVR', ..., 'IIc']
   for name, signal in ecg.items():
       print(f"{name}: {len(signal)} samples, dtype={signal.dtype}")

Generate a PDF from a CSV file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Build an ECG dict from a CSV and render it:

.. code-block:: python

   import pandas as pd
   from ecgtizer.XML2PDF import Write_PDF

   df = pd.read_csv("data/PTB-XL/Original/00129_hr.csv")

   # Map CSV column names to standard lead names
   lead_map = {
       "I": "I", "II": "II", "III": "III",
       "aVR": "AVR", "aVL": "AVL", "aVF": "AVF",
       "V1": "V1", "V2": "V2", "V3": "V3",
       "V4": "V4", "V5": "V5", "V6": "V6",
   }

   ecg = {}
   for csv_col, std_name in lead_map.items():
       ecg[std_name] = df[csv_col].values * 1000  # mV to uV

   # Render as 6x2 PDF
   Write_PDF(ecg, "output/from_csv_type2.pdf", type_of_pdf="type2")

   # Render as 3x4 PDF (requires lead II as rhythm strip)
   Write_PDF(ecg, "output/from_csv_type1.pdf", type_of_pdf="type1",
             lead_IIc=ecg["II"])

Generate a PNG instead of PDF
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Change the output extension to ``.png``:

.. code-block:: python

   from ecgtizer.XML2PDF import Write_PDF

   Write_PDF(ecg, "output/ecg_render.png", type_of_pdf="type2")

Parse a digit string from an XML element
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from ecgtizer.XML2PDF import read_lead

   values = read_lead("120.5 -30.2 abc 450 0")
   print(values)   # [120.5, -30.2, nan, 450.0, 0.0]

The ecg_plot class
^^^^^^^^^^^^^^^^^^

For full control over the ECG rendering:

.. code-block:: python

   from ecgtizer.XML2PDF import ecg_plot

   # Create a plotter with A4 page dimensions
   plotter = ecg_plot(paper_w=210, paper_h=297)

   print(plotter.standard_values)
   # {'y_grid_size': 5, 'x_grid_size': 5, ...}

   print(plotter.lead_index)
   # ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
   #  'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
