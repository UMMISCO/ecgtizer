"""ECGtizer -- convert PDF/image ECGs to digital signals (HL7 aECG XML).

This package provides tools to digitize electrocardiogram recordings from
PDF documents and images, export them in HL7 aECG XML format, complete
partial leads using a deep-learning model, and analyse digitized signals.
"""

from .ecgtizer import ECGtizer
from .analyses import analyse, BlandAltman, scatter_plot, overlap_plot
from .XML2PDF import xml_to_pdf
from .anonymisation import anonymisation

__all__ = [
    "ECGtizer",
    "analyse",
    "BlandAltman",
    "scatter_plot",
    "overlap_plot",
    "xml_to_pdf",
    "anonymisation",
]
