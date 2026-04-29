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


def build_master_player_features():
    tables = load_lahman_tables()

    master = tables["master"]
    batting = tables["batting"]
    salaries = tables["salaries"]
    hof = tables["hof"]

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
    ]]

    df = batting_agg.merge(master_keep, on="playerID", how="left")
    df = df.merge(salaries_agg, on=["playerID", "yearID"], how="left")
    df = df.merge(hof_target, on="playerID", how="left")

    df["inducted"] = df["inducted"].fillna(0).astype(int)
    df["age"] = df["yearID"] - df["birthYear"]

    df["1B"] = df["H"] - df["2B"] - df["3B"] - df["HR"]

    df["BA"] = df["H"] / df["AB"]
    df["OBP_proxy"] = (df["H"] + df["BB"] + df["HBP"]) / (
        df["AB"] + df["BB"] + df["HBP"] + df["SF"]
    )
    df["SLG"] = (df["1B"] + 2 * df["2B"] + 3 * df["3B"] + 4 * df["HR"]) / df["AB"]
    df["OPS_proxy"] = df["OBP_proxy"] + df["SLG"]

    df["HR_rate"] = df["HR"] / df["AB"]
    df["BB_rate"] = df["BB"] / df["AB"]
    df["SO_rate"] = df["SO"] / df["AB"]

    df = df.replace([float("inf"), -float("inf")], pd.NA)

    df = df.sort_values(["playerID", "yearID"])
    df["career_season_number"] = df.groupby("playerID").cumcount() + 1
    df["career_G"] = df.groupby("playerID")["G"].cumsum()
    df["career_AB"] = df.groupby("playerID")["AB"].cumsum()
    df["career_H"] = df.groupby("playerID")["H"].cumsum()
    df["career_HR"] = df.groupby("playerID")["HR"].cumsum()

    df["BA_next"] = df.groupby("playerID")["BA"].shift(-1)
    df["HR_next"] = df.groupby("playerID")["HR"].shift(-1)
    df["power_hitter_next"] = (df["HR_next"] >= 20).astype(int)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_DIR / "master_player_features.parquet", index=False)

    return df


if __name__ == "__main__":
    tables = load_lahman_tables()
    print(summarize_tables(tables))

    master = build_master_player_features()
    print(master.shape)
    print(master.head())