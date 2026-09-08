from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import pandas as pd

from modules.moneycontrol_client import (
    fetch_fundamentals,
    fetch_nav_bundle,
    fetch_nifty100_series,
    fetch_portfolio,
    fetch_risk_metrics,
    fetch_rolling_returns,
    load_fund_master,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SNAPSHOT_DIR = DATA_DIR / "daily_snapshots" / date.today().isoformat()
MASTER_PATH = DATA_DIR / "isin_largecap.xlsx"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    def serialize(value: object) -> object:
        if isinstance(value, pd.DataFrame):
            return value.to_dict(orient="records")
        return str(value)

    path.write_text(json.dumps(payload, default=serialize, indent=2), encoding="utf-8")


def refresh() -> None:
    os.environ["FORCE_API_REFRESH"] = "1"
    fund_master = load_fund_master(str(MASTER_PATH))
    (SNAPSHOT_DIR / "nav").mkdir(parents=True, exist_ok=True)
    (SNAPSHOT_DIR / "rolling_returns").mkdir(parents=True, exist_ok=True)
    (SNAPSHOT_DIR / "portfolio").mkdir(parents=True, exist_ok=True)
    (SNAPSHOT_DIR / "fundamentals").mkdir(parents=True, exist_ok=True)
    (SNAPSHOT_DIR / "risk_metrics").mkdir(parents=True, exist_ok=True)

    benchmark = fetch_nifty100_series()
    benchmark.to_csv(SNAPSHOT_DIR / "nifty100.csv", index=False)

    for fund in fund_master.itertuples(index=False):
        safe_name = "".join(character if character.isalnum() else "_" for character in fund.fund_name).strip("_")
        nav = fetch_nav_bundle(fund.isin)
        nav["fund_nav"].to_csv(SNAPSHOT_DIR / "nav" / f"{safe_name}.csv", index=False)
        for duration in ("1Y", "3Y", "5Y"):
            rolling = fetch_rolling_returns(fund.isin, duration)
            rolling["returns"].to_csv(SNAPSHOT_DIR / "rolling_returns" / f"{safe_name}_{duration}.csv", index=False)
        write_json(SNAPSHOT_DIR / "portfolio" / f"{safe_name}.json", fetch_portfolio(fund.isin))
        write_json(SNAPSHOT_DIR / "fundamentals" / f"{safe_name}.json", fetch_fundamentals(fund.isin))
        write_json(SNAPSHOT_DIR / "risk_metrics" / f"{safe_name}.json", fetch_risk_metrics(fund.isin))
        print(f"refreshed {fund.fund_name}")


if __name__ == "__main__":
    refresh()