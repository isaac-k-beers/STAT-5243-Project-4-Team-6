from __future__ import annotations
from pathlib import Path
import zipfile
import pandas as pd

REQUIRED_KAGGLE_FILES = [
    "Master.csv", "Batting.csv", "Pitching.csv", "Fielding.csv", "HallOfFame.csv"
]
RECOMMENDED_KAGGLE_FILES = [
    "AllstarFull.csv", "AwardsPlayers.csv", "AwardsSharePlayers.csv", "Salaries.csv",
    "Teams.csv", "BattingPost.csv", "PitchingPost.csv", "FieldingOF.csv", "SeriesPost.csv",
    # Extra SABR / Lahman 2025 tables used when the full Lahman release is present.
    "Appearances.csv", "FieldingPost.csv", "TeamsFranchises.csv", "TeamsHalf.csv",
]

def find_csv(data_dir: Path, name: str) -> Path | None:
    direct = data_dir / name
    if direct.exists():
        return direct
    zipped = data_dir / f"{name}.zip"
    if zipped.exists():
        return zipped
    return None

def read_csv_flexible(data_dir: Path, name: str, required: bool = True) -> pd.DataFrame:
    path = find_csv(data_dir, name)
    if path is None:
        if required:
            raise FileNotFoundError(f"Missing required file: {data_dir / name}")
        return pd.DataFrame()
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            csv_members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
            target = next((m for m in csv_members if Path(m).name.lower() == name.lower()), csv_members[0])
            with zf.open(target) as fh:
                return pd.read_csv(fh, low_memory=False)
    return pd.read_csv(path, low_memory=False)

def load_kaggle_tables(data_dir: Path) -> dict[str, pd.DataFrame]:
    tables = {}
    for name in REQUIRED_KAGGLE_FILES:
        tables[Path(name).stem] = read_csv_flexible(data_dir, name, required=True)
    for name in RECOMMENDED_KAGGLE_FILES:
        tables[Path(name).stem] = read_csv_flexible(data_dir, name, required=False)
    return tables
