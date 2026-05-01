"""Part 1 — Advanced Data Acquisition & Preparation.

This script makes the Part 1 workflow explicit and reproducible:
1. Kaggle Baseball Databank CSVs are copied into a stable analysis folder.
2. SABR/Lahman CSVs are downloaded or read from data/raw/sabr_lahman and used
   when they are newer/available. The modern People.csv table is mapped to
   Master.csv so the rest of the pipeline can run unchanged.
3. pybaseball is used for a small, robust supplemental pull. FanGraphs endpoints
   can return 403, so this script also attempts Baseball Savant/ID-lookup pulls.
4. BeautifulSoup scrapes a small Hall of Fame future-eligibles page.
5. A source manifest and data-quality audit tables are saved for the report.

Run from the project root:
    python scripts/01a_data_acquisition_part1.py

The script is designed not to crash if optional internet-based sources are
blocked. It records failures in reports/tables/part1_external_source_status.csv.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import (  # noqa: E402
    DATA_RAW_ANALYSIS,
    DATA_RAW_KAGGLE,
    DATA_RAW_PYBASEBALL,
    DATA_RAW_SABR,
    DATA_RAW_SCRAPED,
    TABLES,
)
from src.data_loader import (  # noqa: E402
    REQUIRED_KAGGLE_FILES,
    RECOMMENDED_KAGGLE_FILES,
    find_csv,
)

SABR_URL = "https://sabr.org/lahman-database/"
HOF_FUTURE_ELIGIBLES_URL = "https://baseballhall.org/hall-of-fame/future-eligibles"
USER_AGENT = "Mozilla/5.0 (compatible; STAT5243Project4/1.0; educational project)"
PROJECT_TABLES = REQUIRED_KAGGLE_FILES + RECOMMENDED_KAGGLE_FILES
SABR_NAME_MAP = {
    "People.csv": "Master.csv",  # current Lahman naming; Kaggle mirror uses Master.csv
    "Master.csv": "Master.csv",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def mkdirs() -> None:
    for p in [DATA_RAW_ANALYSIS, DATA_RAW_SABR, DATA_RAW_PYBASEBALL, DATA_RAW_SCRAPED, TABLES]:
        p.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_zip_csv(src_zip: Path, target_csv_name: str, dst: Path) -> None:
    with zipfile.ZipFile(src_zip) as zf:
        members = [m for m in zf.namelist() if Path(m).name.lower() == target_csv_name.lower()]
        if not members:
            members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        if not members:
            raise FileNotFoundError(f"No CSV found inside {src_zip}")
        with zf.open(members[0]) as src, open(dst, "wb") as out:
            shutil.copyfileobj(src, out)


def find_sabr_download_link() -> str | None:
    """Find a direct CSV/zip/7z link from the SABR Lahman page if possible."""
    response = requests.get(SABR_URL, headers={"User-Agent": USER_AGENT}, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    candidates = []
    for a in soup.find_all("a", href=True):
        text = " ".join(a.get_text(" ", strip=True).split()).lower()
        href = urljoin(SABR_URL, a["href"])
        href_l = href.lower()
        score = 0
        if "comma" in text or "csv" in text:
            score += 3
        if "lahman" in href_l:
            score += 2
        if href_l.endswith((".zip", ".csv", ".7z")):
            score += 3
        if "download" in href_l:
            score += 1
        if score > 0:
            candidates.append((score, text, href))
    candidates.sort(reverse=True)
    for _, _, href in candidates:
        if href.lower().endswith((".zip", ".csv", ".7z")) or "download" in href.lower():
            return href
    return candidates[0][2] if candidates else None


def download_and_extract_sabr(force: bool = False) -> tuple[bool, str]:
    """Download SABR/Lahman if possible. Manual placement also works."""
    existing_csvs = list(DATA_RAW_SABR.rglob("*.csv"))
    if existing_csvs and not force:
        return True, f"SABR/Lahman CSVs already present ({len(existing_csvs)} files)."
    try:
        link = find_sabr_download_link()
        if not link:
            return False, "Could not find a SABR comma-delimited download link automatically."
        out = DATA_RAW_SABR / Path(link.split("?")[0]).name
        if not out.suffix:
            out = DATA_RAW_SABR / "sabr_lahman_download"
        r = requests.get(link, headers={"User-Agent": USER_AGENT}, timeout=60)
        r.raise_for_status()
        out.write_bytes(r.content)
        if out.suffix.lower() == ".zip":
            extract_dir = DATA_RAW_SABR / "latest_extracted"
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(out) as zf:
                zf.extractall(extract_dir)
            return True, f"Downloaded and extracted SABR/Lahman from {link}."
        if out.suffix.lower() == ".csv":
            return True, f"Downloaded one SABR/Lahman CSV from {link}."
        return True, f"Downloaded SABR/Lahman archive to {out}. If it is .7z, extract it manually."
    except Exception as exc:  # noqa: BLE001
        return False, f"SABR/Lahman automatic download failed: {exc}"


def find_sabr_csv_by_project_name(project_name: str) -> Path | None:
    """Locate SABR/Lahman CSV corresponding to a project table name."""
    desired = project_name.lower()
    reverse_aliases = [project_name]
    if project_name == "Master.csv":
        reverse_aliases = ["People.csv", "Master.csv"]
    for alias in reverse_aliases:
        for path in DATA_RAW_SABR.rglob("*.csv"):
            if path.name.lower() == alias.lower():
                return path
    # Some extracts include subfolders or lowercase names.
    for path in DATA_RAW_SABR.rglob("*.csv"):
        if path.name.lower() == desired:
            return path
    return None


def prepare_analysis_input(prefer_sabr: bool = True) -> pd.DataFrame:
    """Create data/raw/analysis_input by reconciling Kaggle and SABR/Lahman."""
    DATA_RAW_ANALYSIS.mkdir(parents=True, exist_ok=True)
    records = []
    for name in PROJECT_TABLES:
        chosen_path = None
        source = None
        sabr_path = find_sabr_csv_by_project_name(name) if prefer_sabr else None
        kaggle_path = find_csv(DATA_RAW_KAGGLE, name)
        if sabr_path is not None:
            chosen_path = sabr_path
            source = "SABR/Lahman"
        elif kaggle_path is not None:
            chosen_path = kaggle_path
            source = "Kaggle Baseball Databank"
        else:
            records.append({
                "table": name,
                "status": "missing",
                "source_used": None,
                "source_path": None,
                "analysis_path": None,
                "rows": None,
                "columns": None,
                "min_year": None,
                "max_year": None,
            })
            continue
        dst = DATA_RAW_ANALYSIS / name
        if chosen_path.suffix.lower() == ".zip":
            copy_zip_csv(chosen_path, name, dst)
        else:
            copy_file(chosen_path, dst)
        # Audit the resulting CSV.
        try:
            df = pd.read_csv(dst, low_memory=False)
            year_cols = [c for c in df.columns if c.lower() in {"yearid", "year_id", "year"}]
            if year_cols:
                y = pd.to_numeric(df[year_cols[0]], errors="coerce")
                min_year, max_year = int(y.min()) if y.notna().any() else None, int(y.max()) if y.notna().any() else None
            else:
                min_year, max_year = None, None
            rows, cols = df.shape
        except Exception:  # noqa: BLE001
            rows = cols = min_year = max_year = None
        records.append({
            "table": name,
            "status": "ready",
            "source_used": source,
            "source_path": str(chosen_path),
            "analysis_path": str(dst),
            "rows": rows,
            "columns": cols,
            "min_year": min_year,
            "max_year": max_year,
        })
    manifest = pd.DataFrame(records)
    manifest.to_csv(TABLES / "part1_source_manifest.csv", index=False)
    return manifest


def scrape_hof_future_eligibles() -> tuple[bool, str]:
    DATA_RAW_SCRAPED.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(HOF_FUTURE_ELIGIBLES_URL, headers={"User-Agent": USER_AGENT}, timeout=30)
        r.raise_for_status()
        (DATA_RAW_SCRAPED / "hof_future_eligibles_page.html").write_text(r.text, encoding="utf-8")
        soup = BeautifulSoup(r.text, "html.parser")
        rows = []
        for tag in soup.find_all(["h1", "h2", "h3", "li", "td", "p", "a"]):
            text = " ".join(tag.get_text(" ", strip=True).split())
            if 4 <= len(text) <= 160 and any(ch.isalpha() for ch in text):
                # Keep informative rows; remove obvious navigation/social text.
                low = text.lower()
                if low not in {"facebook", "twitter", "instagram", "youtube", "search"}:
                    year_match = re.search(r"\b(20\d{2})\b", text)
                    rows.append({"text": text, "year_found": year_match.group(1) if year_match else None})
        df = pd.DataFrame(rows).drop_duplicates()
        df.to_csv(DATA_RAW_SCRAPED / "hof_future_eligibles_scraped_text.csv", index=False)
        return True, f"BeautifulSoup scrape saved {len(df)} text rows."
    except Exception as exc:  # noqa: BLE001
        return False, f"BeautifulSoup scrape failed: {exc}"


def pull_pybaseball_supplement() -> tuple[bool, str]:
    DATA_RAW_PYBASEBALL.mkdir(parents=True, exist_ok=True)
    messages = []
    success_any = False
    try:
        # ID lookup is lightweight and usually more reliable than FanGraphs pages.
        from pybaseball import playerid_lookup
        names = [
            ("trout", "mike"),
            ("kershaw", "clayton"),
            ("pujols", "albert"),
            ("judge", "aaron"),
            ("ichiro", "suzuki"),
        ]
        frames = []
        for last, first in names:
            try:
                lookup = playerid_lookup(last, first)
                lookup["lookup_query"] = f"{first} {last}"
                frames.append(lookup)
            except Exception as exc:  # noqa: BLE001
                messages.append(f"playerid_lookup({first} {last}) failed: {exc}")
        if frames:
            ids = pd.concat(frames, ignore_index=True).drop_duplicates()
            ids.to_csv(DATA_RAW_PYBASEBALL / "pybaseball_playerid_lookup_examples.csv", index=False)
            success_any = True
            messages.append(f"Saved pybaseball ID lookup examples ({len(ids)} rows).")
    except Exception as exc:  # noqa: BLE001
        messages.append(f"pybaseball player ID lookup unavailable: {exc}")

    try:
        # FanGraphs can return 403. Baseball Savant Statcast is a useful fallback.
        from pybaseball import statcast
        sample_windows = [("2015-04-05", "2015-04-05"), ("2024-04-01", "2024-04-01")]
        statcast_df = pd.DataFrame()
        used_window = None
        for start_dt, end_dt in sample_windows:
            try:
                tmp = statcast(start_dt=start_dt, end_dt=end_dt)
                if tmp is not None and len(tmp) > 0:
                    statcast_df = tmp
                    used_window = (start_dt, end_dt)
                    break
            except Exception as exc:  # noqa: BLE001
                messages.append(f"statcast({start_dt}, {end_dt}) failed: {exc}")
        if len(statcast_df) > 0:
            raw_path = DATA_RAW_PYBASEBALL / f"pybaseball_statcast_sample_{used_window[0]}_{used_window[1]}.csv"
            statcast_df.to_csv(raw_path, index=False)
            useful = [c for c in ["game_date", "batter", "pitcher", "events", "description", "release_speed", "launch_speed", "estimated_woba_using_speedangle"] if c in statcast_df.columns]
            statcast_df[useful].to_csv(DATA_RAW_PYBASEBALL / "pybaseball_statcast_sample_trimmed.csv", index=False)
            success_any = True
            messages.append(f"Saved pybaseball Statcast sample ({len(statcast_df)} rows).")
    except Exception as exc:  # noqa: BLE001
        messages.append(f"pybaseball Statcast unavailable: {exc}")

    return success_any, " | ".join(messages) if messages else "No pybaseball sources ran."


def write_status(status_rows: Iterable[dict]) -> None:
    status_df = pd.DataFrame(list(status_rows))
    status_df.to_csv(TABLES / "part1_external_source_status.csv", index=False)
    json_path = DATA_RAW_ANALYSIS.parent / "source_manifest.json"
    json_path.write_text(json.dumps(status_df.to_dict(orient="records"), indent=2), encoding="utf-8")


def main() -> None:
    mkdirs()
    statuses = []

    skip_internet = os.getenv("PROJECT4_SKIP_INTERNET", "0") == "1"

    if skip_internet:
        ok, msg = False, "Internet-based acquisition skipped because PROJECT4_SKIP_INTERNET=1."
    else:
        ok, msg = download_and_extract_sabr(force=False)
    statuses.append({"timestamp_utc": _now(), "source": "SABR/Lahman", "success": ok, "message": msg})
    print(f"SABR/Lahman: {msg}")

    manifest = prepare_analysis_input(prefer_sabr=True)
    print("\nAnalysis input manifest saved to reports/tables/part1_source_manifest.csv")
    print(manifest[["table", "status", "source_used", "rows", "min_year", "max_year"]].to_string(index=False))

    if skip_internet:
        ok, msg = False, "Internet-based BeautifulSoup scrape skipped because PROJECT4_SKIP_INTERNET=1."
    else:
        ok, msg = scrape_hof_future_eligibles()
    statuses.append({"timestamp_utc": _now(), "source": "BeautifulSoup Hall of Fame scrape", "success": ok, "message": msg})
    print(f"\nBeautifulSoup: {msg}")

    if skip_internet:
        ok, msg = False, "Internet-based pybaseball acquisition skipped because PROJECT4_SKIP_INTERNET=1."
    else:
        ok, msg = pull_pybaseball_supplement()
    statuses.append({"timestamp_utc": _now(), "source": "pybaseball", "success": ok, "message": msg})
    print(f"pybaseball: {msg}")

    write_status(statuses)
    print("\nPart 1 acquisition complete.")
    print(f"Use this folder for feature-building: {DATA_RAW_ANALYSIS}")
    print("External-source status saved to reports/tables/part1_external_source_status.csv")


if __name__ == "__main__":
    main()
