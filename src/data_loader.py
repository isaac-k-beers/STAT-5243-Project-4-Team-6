from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def load_csv(name):
    path = RAW_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def load_lahman_tables():
    return {
        "master": load_csv("Master.csv"),
        "batting": load_csv("Batting.csv"),
        "pitching": load_csv("Pitching.csv"),
        "fielding": load_csv("Fielding.csv"),
        "teams": load_csv("Teams.csv"),
        "salaries": load_csv("Salaries.csv"),
        "hof": load_csv("HallOfFame.csv"),
        "awards_players": load_csv("AwardsPlayers.csv"),
        "allstar": load_csv("AllstarFull.csv"),
    }


def summarize_tables(tables):
    rows = []

    for name, df in tables.items():
        rows.append({
            "table": name,
            "rows": df.shape[0],
            "columns": df.shape[1],
            "missing_values": int(df.isna().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum())
        })

    return pd.DataFrame(rows)


def build_master_player_features(min_ab=50):
    tables = load_lahman_tables()

    master = tables["master"]
    batting = tables["batting"]
    salaries = tables["salaries"]
    hof = tables["hof"]

    # Aggregate batting across multiple team stints in the same season
    batting_agg = (
        batting
        .groupby(["playerID", "yearID"], as_index=False)
        .agg({
            "G": "sum",
            "AB": "sum",
            "R": "sum",
            "H": "sum",
            "2B": "sum",
            "3B": "sum",
            "HR": "sum",
            "RBI": "sum",
            "SB": "sum",
            "CS": "sum",
            "BB": "sum",
            "SO": "sum",
            "IBB": "sum",
            "HBP": "sum",
            "SH": "sum",
            "SF": "sum",
            "GIDP": "sum",
        })
    )

    salaries_agg = (
        salaries
        .groupby(["playerID", "yearID"], as_index=False)
        .agg({"salary": "sum"})
    )

    hof_target = (
        hof
        .groupby("playerID", as_index=False)
        .agg(inducted=("inducted", lambda x: int((x == "Y").any())))
    )

    master_keep = master[[
        "playerID", "birthYear", "birthMonth", "birthDay",
        "nameFirst", "nameLast", "weight", "height",
        "bats", "throws", "debut", "finalGame"
    ]].copy()

    df = batting_agg.merge(master_keep, on="playerID", how="left")
    df = df.merge(salaries_agg, on=["playerID", "yearID"], how="left")
    df = df.merge(hof_target, on="playerID", how="left")

    df["inducted"] = df["inducted"].fillna(0).astype(int)

    # Basic cleaning
    df = df[df["yearID"].notna()].copy()
    df["yearID"] = df["yearID"].astype(int)

    df = df[df["AB"].notna()].copy()
    df = df[df["AB"] > 0].copy()

    # Filter out extremely small batting samples
    df = df[df["AB"] >= min_ab].copy()

    # Basic demographic features
    df["age"] = df["yearID"] - df["birthYear"]

    # Era features
    df["era"] = pd.cut(
        df["yearID"],
        bins=[0, 1900, 1972, 1993, 2005, 2100],
        labels=[
            "pre_1900_deadball",
            "pre_dh_era",
            "early_dh_era",
            "steroid_era",
            "modern_era"
        ]
    )

    df["pre_1900_flag"] = (df["yearID"] < 1900).astype(int)
    df["dh_era_flag"] = (df["yearID"] >= 1973).astype(int)
    df["steroid_era_flag"] = df["yearID"].between(1994, 2005).astype(int)

    # Offensive rate features
    df["1B"] = df["H"] - df["2B"] - df["3B"] - df["HR"]

    df["BA"] = df["H"] / df["AB"]

    df["OBP_proxy"] = (df["H"] + df["BB"] + df["HBP"]) / (
        df["AB"] + df["BB"] + df["HBP"] + df["SF"]
    )

    df["SLG"] = (
        df["1B"] + 2 * df["2B"] + 3 * df["3B"] + 4 * df["HR"]
    ) / df["AB"]

    df["OPS_proxy"] = df["OBP_proxy"] + df["SLG"]

    df["HR_rate"] = df["HR"] / df["AB"]
    df["BB_rate"] = df["BB"] / df["AB"]
    df["SO_rate"] = df["SO"] / df["AB"]
    df["RBI_rate"] = df["RBI"] / df["AB"]
    df["SB_rate"] = df["SB"] / df["AB"]

    df = df.replace([float("inf"), -float("inf")], pd.NA)

    # Career features
    df = df.sort_values(["playerID", "yearID"]).copy()

    df["career_season_number"] = df.groupby("playerID").cumcount() + 1
    df["career_longevity"] = df.groupby("playerID")["yearID"].transform("nunique")

    for col in ["G", "AB", "R", "H", "HR", "RBI", "BB", "SO", "SB"]:
        df[f"career_{col}"] = df.groupby("playerID")[col].cumsum()

    # Lagged / next-season targets
    df["BA_next"] = df.groupby("playerID")["BA"].shift(-1)
    df["HR_next"] = df.groupby("playerID")["HR"].shift(-1)
    df["OPS_next"] = df.groupby("playerID")["OPS_proxy"].shift(-1)

    df["power_hitter_next"] = (df["HR_next"] >= 20).astype(int)

    # Modeling-ready indicator
    df["has_next_season_target"] = df["BA_next"].notna().astype(int)

    # Save full cleaned table
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    full_path = PROCESSED_DIR / "master_player_features.parquet"
    df.to_parquet(full_path, index=False)

    # Save modeling subset with valid next-season targets
    model_df = df.dropna(subset=["BA_next", "HR_next", "OPS_next"]).copy()

    model_path = PROCESSED_DIR / "modeling_player_features.parquet"
    model_df.to_parquet(model_path, index=False)

    return df, model_df


if __name__ == "__main__":
    tables = load_lahman_tables()

    print("\nRaw table summary:")
    print(summarize_tables(tables))

    master, model_df = build_master_player_features(min_ab=50)

    print("\nMaster player-year dataset shape:")
    print(master.shape)

    print("\nModeling-ready dataset shape:")
    print(model_df.shape)

    print("\nSample target columns:")
    print(master[[
        "playerID", "yearID", "nameFirst", "nameLast",
        "AB", "BA", "BA_next", "HR", "HR_next", "power_hitter_next"
    ]].head(10))

    print("\nTop missingness rates:")
    print(master.isna().mean().sort_values(ascending=False).head(15))