from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_RAW_KAGGLE = DATA_RAW / "kaggle"
DATA_RAW_SABR = DATA_RAW / "sabr_lahman"
DATA_RAW_PYBASEBALL = DATA_RAW / "pybaseball"
DATA_RAW_SCRAPED = DATA_RAW / "scraped"
# Analysis input is the reconciled folder used by the modeling pipeline.
# It starts from Kaggle and prefers newer SABR/Lahman CSVs when available.
DATA_RAW_ANALYSIS = DATA_RAW / "analysis_input"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS = PROJECT_ROOT / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"
MODELS = PROJECT_ROOT / "models"
APP_DIR = PROJECT_ROOT / "app"

RANDOM_STATE = 42
HOLDOUT_ELIGIBILITY_YEAR = 2000
