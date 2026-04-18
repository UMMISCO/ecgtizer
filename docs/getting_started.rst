Getting Started
===============

System Requirements
-------------------

- Python 3.9 or later
- **poppler** (required by ``pdf2image`` for PDF-to-image conversion)

Install poppler:

.. code-block:: bash

   # macOS
   brew install poppler

   # Ubuntu / Debian
   sudo apt-get install poppler-utils

   # Fedora
   sudo dnf install poppler-utils

Installation
------------

From source:

.. code-block:: bash

   git clone https://github.com/your-org/ecgtizer.git
   cd ecgtizer
   pip install -e .

Development install (includes test and lint tools):

.. code-block:: bash

   pip install -e ".[dev]"
   pre-commit install

To build the documentation locally:

.. code-block:: bash

   pip install -e ".[docs]"
   cd docs
   make html

Quick Start
-----------

Extract ECG signals from a PDF:

.. code-block:: python

   from ecgtizer import ECGtizer

   ecg = ECGtizer(
       file="path/to/ecg.pdf",
       dpi=500,
       extraction_method="fragmented",
       verbose=True,
   )

   # Digitized leads as a dict of numpy arrays
   leads = ecg.extracted_lead

Plot the results:

.. code-block:: python

   ecg.plot()                           # all leads
   ecg.plot(lead="II", save="II.png")   # single lead

Export to HL7 aECG XML:

.. code-block:: python

   ecg.save_xml("output.xml")
