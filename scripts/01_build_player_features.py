from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import DATA_RAW_KAGGLE, DATA_RAW_ANALYSIS, DATA_PROCESSED
from src.data_loader import REQUIRED_KAGGLE_FILES, find_csv
from src.features import build_player_features


def choose_raw_dir():
    """Prefer reconciled Kaggle+SABR analysis_input if Part 1 acquisition created it."""
    if DATA_RAW_ANALYSIS.exists() and all(find_csv(DATA_RAW_ANALYSIS, name) for name in REQUIRED_KAGGLE_FILES):
        return DATA_RAW_ANALYSIS
    return DATA_RAW_KAGGLE


if __name__ == "__main__":
    raw_dir = choose_raw_dir()
    print(f"Building player features from: {raw_dir}")
    df = build_player_features(raw_dir, DATA_PROCESSED)
    print(f"Saved data/processed/player_features_base.csv")
    print(f"Shape: {df.shape[0]:,} players × {df.shape[1]:,} columns")
    print(f"Eligible modeling pool: {int(df['model_eligible_pool'].sum()):,}")
    print(f"Inducted positives in pool: {int(df.loc[df['model_eligible_pool']==1, 'inducted'].sum()):,}")
