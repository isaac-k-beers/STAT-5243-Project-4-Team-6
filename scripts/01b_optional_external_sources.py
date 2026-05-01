"""Alias for the Part 1 acquisition script.

The complete Part 1 acquisition (Kaggle reconciliation, SABR / Lahman
update, pybaseball samples, BeautifulSoup Hall of Fame scrape, and
source-manifest writing) lives entirely in
`scripts/01a_data_acquisition_part1.py`. This file simply runs that
script so older command-line aliases continue to work.

To run Part 1, prefer:

    python scripts/01a_data_acquisition_part1.py
"""
from __future__ import annotations
from pathlib import Path
import runpy

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "01a_data_acquisition_part1.py"
    runpy.run_path(str(target), run_name="__main__")
