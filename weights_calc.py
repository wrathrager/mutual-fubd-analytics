from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


RAW_WEIGHT_DIR = Path("data/weight_change")
OUTPUT_PATH = Path("data/fund_sector_weights.csv")


def clean_fund_name(raw_title: str) -> str:
    fund_name = raw_title.replace("Positions Over Time -", "").strip()
    fund_name = re.sub(r"\s*-\s*Regular\b", "", fund_name, flags=re.IGNORECASE)
    fund_name = re.sub(r"\s*-\s*Reg\b", "", fund_name, flags=re.IGNORECASE)
    fund_name = re.sub(r"\s*\(G\)\s*$", "", fund_name, flags=re.IGNORECASE)
    fund_name = re.sub(r"\s*-\s*\(G\)\s*$", "", fund_name, flags=re.IGNORECASE)
    fund_name = re.sub(r"\s+", " ", fund_name).strip(" -")
    return fund_name


def parse_weight_change_file(file_path: Path) -> pd.DataFrame:
    with file_path.open("r", encoding="latin1") as handle:
        first_line = handle.readline().strip()

    fund_name = clean_fund_name(first_line)
    frame = pd.read_csv(file_path, skiprows=1, encoding="latin1")
    frame = frame.loc[:, ~frame.columns.astype(str).str.contains(r"^Unnamed", na=False)]
    frame.columns = [str(column).strip() for column in frame.columns]

    if "Sector" not in frame.columns:
        raise ValueError(f"Missing Sector column in {file_path.name}")

    date_columns = [
        column
        for column in frame.columns
        if column not in {"Type", "Sector"}
    ]
    melted = frame.melt(
        id_vars=["Sector"],
        value_vars=date_columns,
        var_name="date",
        value_name="weight_pct",
    )
    melted["fund_name"] = fund_name
    melted["sector"] = melted["Sector"].astype(str).str.strip()
    melted["weight_pct"] = pd.to_numeric(
        melted["weight_pct"].replace("-", pd.NA),
        errors="coerce",
    )
    melted["date"] = pd.to_datetime(melted["date"], format="%d-%b-%y", errors="coerce")
    melted = melted.dropna(subset=["date", "weight_pct"])
    melted = melted[melted["sector"].ne("")]

    return melted[["fund_name", "sector", "date", "weight_pct"]]


def build_fund_sector_history(
    raw_dir: Path = RAW_WEIGHT_DIR,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No raw weight files found in {raw_dir}")

    frames = [parse_weight_change_file(file_path) for file_path in files]
    combined = pd.concat(frames, ignore_index=True)
    combined = (
        combined.groupby(["fund_name", "sector", "date"], as_index=False)["weight_pct"]
        .sum()
        .sort_values(["fund_name", "date", "sector"])
        .reset_index(drop=True)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        combined.to_csv(output_path, index=False)
    except PermissionError:
        pass
    return combined


if __name__ == "__main__":
    dataset = build_fund_sector_history()
    print(
        f"Saved {len(dataset)} rows across "
        f"{dataset['fund_name'].nunique()} funds and {dataset['sector'].nunique()} sectors."
    )
