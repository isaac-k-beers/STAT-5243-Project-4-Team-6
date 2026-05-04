from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import DATA_RAW_KAGGLE, DATA_RAW_SABR, DATA_RAW_PYBASEBALL, DATA_RAW_SCRAPED, DATA_RAW_ANALYSIS
from src.data_loader import REQUIRED_KAGGLE_FILES, RECOMMENDED_KAGGLE_FILES, find_csv

print("Data verification")
print("=" * 60)
missing_required = []
for name in REQUIRED_KAGGLE_FILES:
    path = find_csv(DATA_RAW_KAGGLE, name)
    status = "FOUND" if path else "MISSING"
    print(f"{status:8s} Kaggle required    {name}")
    if path is None:
        missing_required.append(name)
for name in RECOMMENDED_KAGGLE_FILES:
    path = find_csv(DATA_RAW_KAGGLE, name)
    status = "FOUND" if path else "optional/missing"
    print(f"{status:16s} Kaggle recommended {name}")

print("\nExternal-source folders")
print("- SABR/Lahman folder:", DATA_RAW_SABR, f"({len(list(DATA_RAW_SABR.rglob('*.csv'))) if DATA_RAW_SABR.exists() else 0} CSV files)")
print("- pybaseball folder:", DATA_RAW_PYBASEBALL, f"({len(list(DATA_RAW_PYBASEBALL.glob('*.csv'))) if DATA_RAW_PYBASEBALL.exists() else 0} CSV files)")
print("- BeautifulSoup scraped folder:", DATA_RAW_SCRAPED, f"({len(list(DATA_RAW_SCRAPED.glob('*'))) if DATA_RAW_SCRAPED.exists() else 0} files)")
print("- Reconciled analysis input folder:", DATA_RAW_ANALYSIS)

if DATA_RAW_ANALYSIS.exists():
    missing_analysis = [name for name in REQUIRED_KAGGLE_FILES if find_csv(DATA_RAW_ANALYSIS, name) is None]
    if missing_analysis:
        print("  analysis_input exists but is missing required files:", missing_analysis)
    else:
        print("  analysis_input is ready and will be used by scripts/01_build_player_features.py")

print("\nNotes:")
if find_csv(DATA_RAW_ANALYSIS, "Appearances.csv") is not None:
    print("- Appearances.csv is available in analysis_input and is used as the canonical "
          "source for primary-position construction (preferred over Fielding.csv).")
else:
    print("- Appearances.csv is optional. When unavailable, primary positions are inferred "
          "from Fielding.csv as a fallback.")
print("- Run scripts/01a_data_acquisition_part1.py to create the Kaggle+SABR reconciled analysis_input and optional pybaseball/BeautifulSoup outputs.")
if missing_required:
    raise SystemExit(f"Missing required Kaggle files: {missing_required}")
print("\nVerification passed.")
