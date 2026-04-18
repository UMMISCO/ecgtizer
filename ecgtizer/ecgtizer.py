"""Main ECGtizer class for PDF/image ECG digitization.

Provides the :class:`ECGtizer` entry point that orchestrates the full
pipeline: PDF-to-image conversion, noise detection, text extraction,
track segmentation, waveform extraction, lead calibration, and optional
deep-learning completion.
"""

from __future__ import annotations

# Modules
from .PDF2XML import (
    convert_PDF2image,
    check_noise_type,
    text_extraction,
    tracks_extraction,
    lead_extraction,
    lead_cutting,
)
from .PDF2XML_mod import plot_function, write_xml, plot_overlay
from .completion import completion_
import cv2
import logging
from typing import Callable

import numpy as np
import time

logger = logging.getLogger(__name__)


class ECGtizer:
    """Digitize ECG recordings from PDF documents or images.

    Orchestrates the full extraction pipeline: PDF-to-image conversion,
    noise/format detection, text masking, track segmentation, waveform
    digitization and lead calibration.

    Parameters
    ----------
    file : str
        Path to the input file (PDF, PNG, JPG, or JPEG).
    dpi : int
        Resolution in dots per inch for the PDF-to-image conversion.
    Callback : Callable or None, optional
        Progress callback function. Called with status strings during
        each pipeline step. Defaults to ``None``.
    extraction_method : str, optional
        Waveform extraction algorithm: ``"lazy"``, ``"full"`` or
        ``"fragmented"``. Defaults to ``"full"``.
    typ : str, optional
        Force a specific ECG format (e.g. ``"classic"``, ``"kardia"``).
        When empty, the format is auto-detected. Defaults to ``""``.
    verbose : bool, optional
        Log timing information for each pipeline step.
        Defaults to ``False``.
    DEBUG : bool, optional
        Show intermediate debug plots. Defaults to ``False``.

    Attributes
    ----------
    extracted_lead : dict[str, numpy.ndarray]
        Digitized leads keyed by name (e.g. ``"I"``, ``"II"``, ``"V1"``).
    TYPE : str
        Detected or forced ECG format.
    good : bool
        ``False`` when the PDF could not be converted.
    """

    def __init__(
        self,
        file: str,
        dpi: int,
        Callback: Callable | None = None,
        extraction_method: str = "full",
        typ: str = "",
        verbose: bool = False,
        DEBUG: bool = False,
    ) -> None:
        ### Variables ###
        self.file = file
        self.typ = typ
        self.dpi = dpi
        self.good = True
        self.extraction_method = extraction_method

        ### "Constant" ###
        self.page = 1
        self.extracted_lead = np.zeros((1,))

        self.table_parameters = {
            "hour": "unknow",
            "day": "unknow",
            "month": "unknow",
            "year": "unknow",
            "Scale_x": "10",
            "Scale_y": "25",
            "low_freq": "unknow",
            "high_freq": "unknow",
            "BPM": "unknow",
            "Inter PR (ms)": "unknow",
            "Dur.QRS (ms)": "unknow",
            "QT (ms)": "unknow",
            "QTc (ms)": "unknow",
            "Axe P": "unknow",
            "Axe R": "unknow",
            "Axe T": "unknow",
            "Moy RR (ms)": "unknow",
            "QTcB (ms)": "unknow",
            "QTcF (ms)": "unknow",
            "Rythme": "unknow",
            "ECG": "unknow",
            "Age": "unknow",
            "sex": "unknow",
            "other_information": "unknow",
        }

        ### All dictionary where pre-processing images are stored ###
        self.all_image = []
        self.all_image_clean = []
        self.dic_tracks = []
        self.dic_tracks_clean = []
        self.dic_tracks_ex = []
        self.dic_image_bin = []
        self.df_patient = []
        self.dic_tracks_ex_not_scale = []
        self.variance = []

        ### Convert PDF files to image ###
        ext = file.lower().rsplit(".", 1)[-1] if "." in file else ""
        if ext == "pdf":
            if verbose:
                logger.info("Conversion PDF in image...")
                start = time.time()
            if Callback is not None:
                Callback("\n")
                Callback("--- Conversion PDF in image : ", end="")
                start = time.time()
            images, page_number, _ = convert_PDF2image(file, DPI=dpi)
            if not _:
                self.good = False
                return None
            self.all_image = images
            if verbose:
                logger.info("Conversion PDF in image: OK (%.2fs)", time.time() - start)
            if Callback is not None:
                Callback("\t\t\tOK (" + str(round(time.time() - start, 2)) + "sec) \n")
            page = 0
        elif ext in ("png", "jpg", "jpeg"):
            if verbose:
                logger.info("Open Image...")
                start = time.time()
            images = [cv2.imread(file)]
            page = 0
            if verbose:
                logger.info("Open Image: OK (%.2fs)", time.time() - start)
        else:
            logger.error("Unsupported file format: .%s", ext)
            self.good = False
            return

        ### Convert all images ###
        for image in images:
            if page > 0 and verbose:
                logger.info("---")

            ### a/ Check the image type ###
            if verbose:
                logger.info("Check Quality and Type of image...")
                start = time.time()
            if Callback is not None:
                Callback("--- Check Quality and Type of image : ", end="")
                start = time.time()
            self.image = np.array(image)
            TYPE, NOISE = check_noise_type(self.image, dpi, DEBUG)
            if self.typ != "":
                TYPE = self.typ
            # Kardia Format is particular
            if TYPE.lower() == "kardia":
                if page_number > 1:
                    FORMAT = "multilead"
                else:
                    FORMAT = "unilead"
            else:
                FORMAT = ""
            if verbose:
                logger.info("Check Quality and Type of image: OK (%.2fs)", time.time() - start)
            if Callback is not None:
                Callback("\t\tOK (" + str(round(time.time() - start, 2)) + "sec) \n")

            ### b/ Check the image type ###
            if verbose:
                logger.info("TYPE: %s", TYPE)
                logger.info("Extract all the text from the image...")
                start = time.time()
            if Callback is not None:
                Callback("--- Extract all the text from the image : ", end="")
                start = time.time()
            # image_clean, df = text_extraction(self.image,page, dpi, NOISE, TYPE, DEBUG = DEBUG)
            image_clean = text_extraction(self.image, page, dpi, NOISE, TYPE, DEBUG=DEBUG)
            # if page == 0:
            #    self.df_patient = df
            image = image_clean
            if verbose:
                logger.info("Extract all the text from the image: OK (%.2fs)", time.time() - start)
            if Callback is not None:
                Callback("\tOK (" + str(round(time.time() - start, 2)) + "sec) \n")

            ### c/ Extract each tracks ###
            if verbose:
                logger.info("Detect tracks position...")
                start = time.time()
            if Callback is not None:
                Callback("--- Detect tracks position : ", end="")
                start = time.time()
            dic_tracks, varianceh, variancev = tracks_extraction(
                self.image, TYPE, dpi, FORMAT, DEBUG=DEBUG, NOISE=NOISE
            )
            self.varianceh = varianceh
            self.variancev = variancev
            self.dic_tracks = dic_tracks
            if verbose:
                logger.info("Detect tracks position: OK (%.2fs)", time.time() - start)
            if Callback is not None:
                Callback("\t\t\tOK (" + str(round(time.time() - start, 2)) + "sec) \n")

            ### c/ Clean the tracks from outliers ###
            # if verbose:
            #    print("--- Clean tracks : ", end='')
            #    start = time.time()
            # if TYPE.lower() != 'kardia':
            #    clean_tracks(dic_tracks,TYPE, NOISE = NOISE, DEBUG = DEBUG )
            #    self.dic_tracks_clean = clean_tracks
            # if verbose:
            #    print("\t\t\t\tOK ("+str(round(start - time.time(), 2)) + "sec) \n")

            ### d/ Convert the tracks in digital format ###
            if verbose:
                logger.info("Tracks extraction...")
                start = time.time()
            if Callback is not None:
                Callback("--- Tracks extraction : ", end="")
                start = time.time()
            dic_tracks_ex, image_bin, dic_tracks_ex_not_scale = lead_extraction(
                dic_tracks, extraction_method, TYPE, NOISE=NOISE, DEBUG=DEBUG
            )
            # dic_tracks_ex = lead_extraction(dic_tracks,TYPE, NOISE = NOISE, DEBUG = DEBUG )
            self.dic_tracks_ex = dic_tracks_ex
            self.dic_tracks_ex_not_scale = dic_tracks_ex_not_scale
            if verbose:
                logger.info("Tracks extraction: OK (%.2fs)", time.time() - start)
            if Callback is not None:
                Callback("\t\t\tOK (" + str(round(time.time() - start, 2)) + "sec) \n")

            if verbose:
                logger.info("Lead detection...")
                start = time.time()
            if Callback is not None:
                Callback("--- Lead detection : ", end="")
                start = time.time()
            dic_lead = lead_cutting(
                dic_tracks_ex, dpi, TYPE, FORMAT, page, NOISE=NOISE, DEBUG=DEBUG, dic_image_bin=image_bin
            )
            if verbose:
                logger.info("Lead detection: OK (%.2fs)", time.time() - start)
            if Callback is not None:
                Callback("\t\t\t\tOK (" + str(round(time.time() - start, 2)) + "sec) \n")

            if TYPE.lower() == "kardia" and FORMAT.lower() == "multilead":
                if page == 0:
                    extracted_lead = dic_lead
                else:
                    for k in dic_lead:
                        pixel_zero = extracted_lead["ref"][0]
                        f = extracted_lead["ref"][1]
                        dic_lead[k] = ((pixel_zero - dic_lead[k]) / f) * 1000
                        extracted_lead[k] = np.concatenate([extracted_lead[k], dic_lead[k]])
                page += 1

            elif page == 0 and TYPE.lower() != "classic" and FORMAT.lower() != "multilead":
                extracted_lead = {}
                extracted_lead["all"] = dic_lead
                page += 1

            elif TYPE.lower() != "classic":
                extracted_lead["all"] = np.concatenate((np.array(extracted_lead["all"]), np.array(dic_lead)))
                page += 1

            self.dic_image_bin.append(image_bin)

        if TYPE.lower() == "classic":
            self.extracted_lead = dic_lead
        else:
            self.extracted_lead = extracted_lead

        self.dic_tracks = dic_tracks
        self.TYPE = TYPE

    def plot(
        self,
        lead: str = "",
        begin: int = 0,
        end: str | int = "inf",
        c: str | None = None,
        save: str | bool = False,
        transparent: bool = False,
        completion: bool = False,
    ) -> None:
        """Plot extracted (or completed) ECG leads.

        Parameters
        ----------
        lead : str, optional
            Name of a single lead to plot (e.g. ``"II"``). When empty,
            all leads are plotted in a grid layout.
        begin : int, optional
            Start sample index. Defaults to ``0``.
        end : str or int, optional
            End sample index, or ``"inf"`` for the full signal.
        c : str or None, optional
            Matplotlib color string.
        save : str or bool, optional
            File path to save the figure, or ``False`` to skip saving.
        transparent : bool, optional
            Save the figure with a transparent background.
        completion : bool, optional
            If ``True``, plot the completed leads instead of the
            raw extracted leads. Requires :meth:`completion` to have
            been called first.
        """
        if not completion:
            plot_function(
                lead_all=self.extracted_lead, lead=lead, b=begin, e=end, c=c, save=save, transparent=transparent
            )
        else:
            plot_function(
                lead_all=self.extracted_lead_comp, lead=lead, b=begin, e=end, c=c, save=save, transparent=transparent
            )

    def plot_over(self) -> None:
        """Overlay extracted waveforms on the original ECG image.

        Displays the source image in grayscale with the digitized
        waveform traces drawn in red for visual verification.
        """
        plot_overlay(lead=self.dic_tracks_ex_not_scale, image=self.image, piqueh=self.varianceh, piquev=self.variancev)

    def save_xml(self, save: str, num_version: str = "0.0", date_version: str = "17.O4.2023") -> None:
        """Export the extracted leads as an HL7 aECG XML file.

        Parameters
        ----------
        save : str
            Output file path for the XML document.
        num_version : str, optional
            Software version string embedded in the XML.
        date_version : str, optional
            Version date string embedded in the XML.
        """
        write_xml(
            matrix=self.extracted_lead,
            path_out=save,
            TYPE=self.TYPE,
            table=self.table_parameters,
            num_version=num_version,
            date_version=date_version,
        )

    def completion(self, path_model: str, device: str) -> None:
        """Complete partial leads to full 10-second recordings.

        Uses a pre-trained PyTorch autoencoder to extend leads that
        cover only 2.5 s or 5 s to the full 10 s duration. The
        completed leads are stored in ``self.extracted_lead_comp``.

        Parameters
        ----------
        path_model : str
            Path to the ``.pth`` model weights file.
        device : str
            PyTorch device string (e.g. ``"cpu"`` or ``"cuda"``).
        """
        self.extracted_lead_comp = completion_(ecg=self.extracted_lead, path_model=path_model, device=device)
