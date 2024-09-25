import sys
sys.path.append("ecgtizer/")

from ecgtizer import ECGtizer
from analyses import analyse, BlandAltman, scatter_plot, overlap_plot
#from anonymisation import anonymisation
from XML2PDF import xml_to_pdf

import argparse


def main(path_input, dpi, method, verbose, path_out):
    """
    Digitizes ECGs from PDF files and saves them in XML format.

    Args:
        path_input (str): Path to the input PDF file containing ECG data.
        dpi (int): DPI (dots per inch) resolution for the input PDF.
        verbose (bool): If True, enable verbose mode to print progress information.
        debug (bool): If True, enable debug mode for additional debugging information.
        path_out (str): Path to save the extracted XML file.

    Returns:
        None
    """
    print(method)
    # Instantiate ECGtizer object with input parameters
    ecg_extracted = ECGtizer (path_input, dpi, method, verbose = verbose, DEBUG = False)

    if ecg_extracted.good != False:
        # Save extracted data to XML file
        ecg_extracted.save_xml(path_out)

# Checks if the script is executed as the main file
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ECG data from PDF files and save it in XML format.")

    # Add arguments to the parser
    parser.add_argument("path_input", type=str, help="Path to the input file.")
    parser.add_argument("dpi", type=int, help="Dots per inch (DPI) of the input image.")
    parser.add_argument("method", type=str, help="Method to extract ECG data.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode to print progress information.")
    parser.add_argument("path_out", type=str, help="Path to save the extracted XML file.")

    # Parse command-line arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args.path_input, args.dpi, args.method, args.verbose, args.path_out)