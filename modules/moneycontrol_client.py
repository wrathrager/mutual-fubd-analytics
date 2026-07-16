from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st


GRAPH_DURATIONS = ["10Y", "7Y", "5Y", "3Y", "1Y", "6M", "3M", "1M"]
ROLLING_DURATIONS = ["5Y", "3Y", "1Y", "6M", "3M"]
CATEGORY_SERIES_ISIN = "INCA000001"
SCORING_CUTOFF_DATE = pd.Timestamp("2026-05-31")
BASE_DIR = Path(__file__).resolve().parent.parent
PRICES_PATH = BASE_DIR / "PRICES.xlsx"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json,text/plain,*/*",
}

REQUEST_MIN_INTERVAL_SECONDS = float(os.getenv("REQUEST_MIN_INTERVAL_SECONDS", "1.5"))
_LAST_REQUEST_AT: float | None = None

HEADERS.update({
    "Referer": "https://www.moneycontrol.com/",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
})


def should_throttle(last_request_at: float | None, now: float, min_interval: float) -> bool:
    if last_request_at is None:
        return False
    return now - last_request_at < min_interval


def normalize_isin(value: str) -> str:
    return str(value).strip().rstrip("/")


def normalize_fund_name(value: str) -> str:
    return " ".join(
        str(value)
        .replace("(G)", "")
        .replace("Regular", "")
        .replace("-", " ")
        .replace("(", " ")
        .replace(")", " ")
        .split()
    ).strip().lower()


def coarse_fund_key(value: str) -> str:
    normalized = normalize_fund_name(value)
    words = normalized.split()
    return " ".join(words[:2]) if len(words) >= 2 else normalized


@st.cache_data(show_spinner=False, ttl=21600)
def load_fund_master(path: str) -> pd.DataFrame:
    frame = pd.read_excel(path)
    frame.columns = [str(column).strip() for column in frame.columns]
    rename_map = {
        next(column for column in frame.columns if "fund" in column.lower()): "fund_name",
        next(column for column in frame.columns if "isin" in column.lower()): "isin",
    }
    frame = frame.rename(columns=rename_map)[["fund_name", "isin"]].copy()
    frame["fund_name"] = frame["fund_name"].astype(str).str.strip()
    frame["isin"] = frame["isin"].map(normalize_isin)
    frame["fund_key"] = frame["fund_name"].map(normalize_fund_name)
    frame["coarse_fund_key"] = frame["fund_name"].map(coarse_fund_key)
    return frame.sort_values("fund_name").reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=21600)
def fetch_json(url: str) -> dict[str, Any]:
    global _LAST_REQUEST_AT

    now = time.time()
    if should_throttle(_LAST_REQUEST_AT, now, REQUEST_MIN_INTERVAL_SECONDS):
        time.sleep(REQUEST_MIN_INTERVAL_SECONDS)

    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and payload.get("success") == 0:
        raise ValueError(f"API returned success=0 for {url}")

    _LAST_REQUEST_AT = time.time()
    return payload


def _pick_duration(
    isin: str,
    durations: list[str],
    url_factory,
    has_data_factory=None,
) -> str:
    for duration in durations:
        payload = fetch_json(url_factory(isin, duration))
        if has_data_factory is not None:
            if has_data_factory(payload, isin):
                return duration
            continue
        data = payload.get("data", {})
        if isinstance(data, dict):
            rows = data.get(isin)
            if isinstance(rows, list) and rows:
                return duration
        elif isinstance(data, list) and data:
            return duration
    raise ValueError(f"No usable data found for {isin} across durations: {durations}")


def _coerce_timeseries(
    rows: list[dict[str, Any]],
    date_column: str,
    value_column: str,
) -> pd.DataFrame:
    frame = pd.DataFrame(rows).copy()
    if frame.empty:
        return pd.DataFrame(columns=["date", "value"])
    frame["date"] = pd.to_datetime(frame[date_column], errors="coerce")
    frame["value"] = pd.to_numeric(frame[value_column], errors="coerce")
    frame = frame.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)
    return frame[["date", "value"]]


@st.cache_data(show_spinner=False, ttl=21600)
def fetch_nav_bundle(isin: str, requested_duration: str | None = None) -> dict[str, Any]:
    isin = normalize_isin(isin)
    def build_url(fund_isin: str, duration: str) -> str:
        return (
            "https://api.moneycontrol.com/swiftapi/v1/mutualfunds/graph-nav"
            f"?&isin={fund_isin},{CATEGORY_SERIES_ISIN}&dur={duration}&deviceType=W&responseType=json"
        )

    duration = requested_duration or _pick_duration(isin, GRAPH_DURATIONS, build_url)
    payload = fetch_json(build_url(isin, duration))
    data = payload["data"]
    fund_series = _coerce_timeseries(data.get(isin, []), "navDate", "navValueAdjusted")
    category_key = next(
        (
            key for key in data.keys()
            if key not in {isin, "availPeriod"}
        ),
        None,
    )
    category_series = _coerce_timeseries(data.get(category_key, []), "navDate", "navValueAdjusted") if category_key else pd.DataFrame(columns=["date", "value"])
    return {
        "duration": duration,
        "available_periods": list(data.get("availPeriod", [])),
        "fund_nav": fund_series,
        "category_nav": category_series,
        "category_key": category_key,
    }


@st.cache_data(show_spinner=False, ttl=21600)
def fetch_rolling_returns(isin: str, requested_duration: str | None = None) -> dict[str, Any]:
    isin = normalize_isin(isin)
    def build_url(fund_isin: str, duration: str) -> str:
        return (
            "https://api.moneycontrol.com/swiftapi/v1/mutualfunds/rolling-returns-new"
            f"?&isin={fund_isin}&dur={duration}&deviceType=W&responseType=json"
        )

    duration = requested_duration or _pick_duration(
        isin,
        ROLLING_DURATIONS,
        build_url,
        has_data_factory=lambda payload, _isin: bool(payload.get("data", {}).get("returns")),
    )
    payload = fetch_json(build_url(isin, duration))
    data = payload["data"]
    returns = pd.DataFrame(data.get("returns", [])).copy()
    if returns.empty:
        returns = pd.DataFrame(columns=["start_date", "end_date", "abs_val", "cagr_val"])
    else:
        returns["start_date"] = pd.to_datetime(returns["start_date"], errors="coerce")
        returns["end_date"] = pd.to_datetime(returns["end_date"], errors="coerce")
        returns["abs_val"] = pd.to_numeric(returns["abs_val"], errors="coerce")
        returns["cagr_val"] = pd.to_numeric(returns["cagr_val"], errors="coerce")
        returns = returns.dropna(subset=["start_date", "end_date"]).sort_values("end_date").reset_index(drop=True)
    return {
        "duration": duration,
        "available_periods": list(data.get("availPeriod", [])),
        "returns": returns,
    }


@st.cache_data(show_spinner=False, ttl=21600)
def fetch_performance(isin: str) -> dict[str, Any]:
    isin = normalize_isin(isin)
    payload = fetch_json(
        "https://api.moneycontrol.com/swiftapi/v1/mutualfunds/performance"
        f"?&isin={isin}&section=annualised&deviceType=W&responseType=json"
    )
    rows = payload.get("data", [])
    return rows[0] if rows else {}


@st.cache_data(show_spinner=False, ttl=21600)
def fetch_portfolio(isin: str) -> dict[str, Any]:
    isin = normalize_isin(isin)
    payload = fetch_json(
        "https://api.moneycontrol.com/swiftapi/v1/mutualfunds/portfolio"
        f"?isin={isin}&deviceType=W&responseType=json"
    )
    items = payload.get("data", [])
    summary = items[0] if len(items) > 0 else {}
    holdings = items[1].get("stock_holding", []) if len(items) > 1 else []
    holdings_frame = pd.DataFrame(holdings).copy()
    if not holdings_frame.empty:
        for column in ["value", "weighting"]:
            if column in holdings_frame.columns:
                holdings_frame[column] = pd.to_numeric(holdings_frame[column], errors="coerce")
    return {
        "summary": summary,
        "holdings": holdings_frame,
    }


@st.cache_data(show_spinner=False, ttl=21600)
def fetch_fundamentals(isin: str) -> dict[str, Any]:
    isin = normalize_isin(isin)
    payload = fetch_json(
        "https://api.moneycontrol.com/swiftapi/v1/mutualfunds/fundamentals"
        f"?isin={isin}&deviceType=W&responseType=json"
    )
    return payload.get("data", {})


@st.cache_data(show_spinner=False, ttl=21600)
def fetch_risk_metrics(isin: str) -> dict[str, Any]:
    isin = normalize_isin(isin)
    payload = fetch_json(
        "https://api.moneycontrol.com/swiftapi/v1/mutualfunds/risk-metrics"
        f"?isin={isin}&deviceType=W&responseType=json"
    )
    return payload.get("data", {})


def _load_local_benchmark_series() -> pd.DataFrame:
    if not PRICES_PATH.exists():
        return pd.DataFrame(columns=["date", "value"])

    try:
        frame = pd.read_excel(PRICES_PATH)
    except Exception:
        return pd.DataFrame(columns=["date", "value"])

    if frame.empty:
        return pd.DataFrame(columns=["date", "value"])

    if "DATE" in frame.columns:
        date_column = "DATE"
    elif "date" in frame.columns:
        date_column = "date"
    else:
        return pd.DataFrame(columns=["date", "value"])

    benchmark_candidates: list[tuple[str, int, pd.Series]] = []
    for column in frame.columns:
        if str(column).strip().lower() == str(date_column).strip().lower():
            continue
        series = pd.to_numeric(frame[column], errors="coerce")
        non_null_count = int(series.notna().sum())
        if non_null_count >= 30:
            benchmark_candidates.append((str(column), non_null_count, series))

    if not benchmark_candidates:
        return pd.DataFrame(columns=["date", "value"])

    preferred_names = ["NIFTY 100", "NIFTY100"]
    preferred_choice = None
    for preferred_name in preferred_names:
        for column_name, _, _ in benchmark_candidates:
            if column_name.upper() == preferred_name.upper():
                preferred_choice = column_name
                break
        if preferred_choice is not None:
            break

    if preferred_choice is None:
        return pd.DataFrame(columns=["date", "value"])

    selected_series = next(series for name, _, series in benchmark_candidates if name == preferred_choice)
    local_frame = pd.DataFrame(
        {
            "date": pd.to_datetime(frame[date_column], errors="coerce"),
            "value": selected_series,
        }
    )
    return local_frame.dropna(subset=["date", "value"])[["date", "value"]].sort_values("date").reset_index(drop=True)


def _fetch_yahoo_nifty100_series() -> pd.DataFrame:
    try:
        response = requests.get(
            "https://query1.finance.yahoo.com/v8/finance/chart/%5ECNX100?range=10y&interval=1d",
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json,text/plain,*/*",
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        result = payload.get("chart", {}).get("result", [])
        if not result:
            return pd.DataFrame(columns=["date", "value"])
        first_result = result[0]
        timestamps = first_result.get("timestamp", [])
        closes = (
            first_result.get("indicators", {})
            .get("quote", [{}])[0]
            .get("close", [])
        )
        if not timestamps or not closes:
            return pd.DataFrame(columns=["date", "value"])
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None).normalize(),
                "value": pd.to_numeric(closes, errors="coerce"),
            }
        )
        return frame.dropna(subset=["date", "value"])[["date", "value"]].sort_values("date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["date", "value"])


@st.cache_data(show_spinner=False, ttl=21600)
def fetch_nifty100_series() -> pd.DataFrame:
    try:
        payload = fetch_json(
            "https://www.moneycontrol.com/techmvc/responsive_api/indices/get_graph_data/"
            "?section=compare_another_index&ind_id=9&another_ind_id=23&range=10yr&classic=true"
        )
    except Exception:
        yahoo_frame = _fetch_yahoo_nifty100_series()
        if not yahoo_frame.empty:
            return yahoo_frame
        return _load_local_benchmark_series()

    rows = payload.get("first", [])
    frame = pd.DataFrame(rows).copy()
    if frame.empty:
        yahoo_frame = _fetch_yahoo_nifty100_series()
        if not yahoo_frame.empty:
            return yahoo_frame
        return _load_local_benchmark_series()
    frame["date"] = pd.to_datetime(frame["time"], format="%d %b %Y", errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    normalized = frame.dropna(subset=["date", "value"])[["date", "value"]].sort_values("date").reset_index(drop=True)
    if normalized.empty:
        yahoo_frame = _fetch_yahoo_nifty100_series()
        if not yahoo_frame.empty:
            return yahoo_frame
        return _load_local_benchmark_series()
    return normalized


def read_json_cache(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return pd.read_json(path, typ="series").to_dict()
