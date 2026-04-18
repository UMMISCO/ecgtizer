"""Command-line interface for ECGtizer.

Usage
-----
    ecgtizer input.pdf 500 fragmented output.xml
    ecgtizer input.pdf 500 fragmented output.xml --verbose
    ecgtizer input.png 300 full output.xml --type kardia
"""

from __future__ import annotations

import argparse
import logging
import sys

from .ecgtizer import ECGtizer


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``ecgtizer`` command."""
    parser = argparse.ArgumentParser(
        prog="ecgtizer",
        description="Digitize ECG from PDF/image and save as HL7 aECG XML.",
    )
    parser.add_argument("input", help="Path to the input PDF, PNG, or JPEG file.")
    parser.add_argument("dpi", type=int, help="Resolution in dots per inch (e.g. 300, 500).")
    parser.add_argument(
        "method",
        choices=["lazy", "full", "fragmented"],
        help="Extraction algorithm: lazy, full, or fragmented.",
    )
    parser.add_argument("output", help="Path for the output XML file.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print progress information.")
    parser.add_argument(
        "--type", "-t", dest="typ", default="", help="Force ECG format (classic, kardia, wellue, apple)."
    )
    args = parser.parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    ecg = ECGtizer(
        args.input,
        dpi=args.dpi,
        extraction_method=args.method,
        typ=args.typ,
        verbose=args.verbose,
    )

    if not ecg.good:
        print(f"Error: extraction failed for {args.input}", file=sys.stderr)
        return 1

    ecg.save_xml(args.output)
    if args.verbose:
        leads = list(ecg.extracted_lead.keys()) if isinstance(ecg.extracted_lead, dict) else ["all"]
        print(f"Saved {len(leads)} leads to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
