from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .moneycontrol_client import SCORING_CUTOFF_DATE


BUILT_IN_SCENARIOS = {
    "2014 Election": ("2014-04-01", "2014-05-30"),
    "2018 Indian Crisis": ("2018-09-01", "2018-10-31"),
    "Demonetisation": ("2016-11-08", "2016-12-30"),
    "Lockdown": ("2020-02-15", "2020-06-30"),
}


def safe_float(value: Any) -> float | None:
    try:
        if value in (None, "", "NA", "N/A"):
            return None
        return float(str(value).replace("%", "").replace(",", ""))
    except (TypeError, ValueError):
        return None


def filter_to_cutoff(frame: pd.DataFrame, cutoff_date: pd.Timestamp = SCORING_CUTOFF_DATE) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return frame.loc[frame["date"] <= cutoff_date].copy()


def _value_on_or_before(frame: pd.DataFrame, target_date: pd.Timestamp) -> tuple[pd.Timestamp, float] | None:
    eligible = frame.loc[frame["date"] <= target_date].sort_values("date")
    if eligible.empty:
        return None
    row = eligible.iloc[-1]
    return pd.Timestamp(row["date"]), float(row["value"])


def calculate_cagr(nav_frame: pd.DataFrame, years: int, as_of: pd.Timestamp | None = None) -> float | None:
    if nav_frame.empty:
        return None
    frame = nav_frame.copy().sort_values("date")
    end_date = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(frame["date"].max())
    end_point = _value_on_or_before(frame, end_date)
    if end_point is None:
        return None
    actual_end_date, end_value = end_point
    target_start = actual_end_date - pd.DateOffset(years=years)
    start_point = _value_on_or_before(frame, target_start)
    if start_point is None:
        return None
    actual_start_date, start_value = start_point
    day_count = (actual_end_date - actual_start_date).days
    if day_count <= 0 or start_value <= 0 or end_value <= 0:
        return None
    return ((end_value / start_value) ** (365.25 / day_count) - 1) * 100


def build_monthly_returns(nav_frame: pd.DataFrame) -> pd.DataFrame:
    if nav_frame.empty:
        return pd.DataFrame(columns=["date", "monthly_return_pct", "year", "month"])
    monthly = (
        nav_frame.set_index("date")["value"]
        .resample("ME")
        .last()
        .to_frame("nav")
        .reset_index()
    )
    monthly["monthly_return_pct"] = monthly["nav"].pct_change() * 100
    monthly["year"] = monthly["date"].dt.year
    monthly["month"] = monthly["date"].dt.month
    monthly["month_name"] = monthly["date"].dt.strftime("%b").str.upper()
    return monthly.dropna(subset=["monthly_return_pct"]).reset_index(drop=True)


def build_monthly_heatmap(monthly_returns: pd.DataFrame) -> pd.DataFrame:
    if monthly_returns.empty:
        return pd.DataFrame()
    heatmap = monthly_returns.pivot(index="year", columns="month_name", values="monthly_return_pct")
    month_order = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    heatmap = heatmap.reindex(columns=month_order)
    heatmap["YEAR"] = monthly_returns.groupby("year")["monthly_return_pct"].apply(
        lambda values: ((1 + values / 100).prod() - 1) * 100
    )
    return heatmap.sort_index(ascending=False)


def build_yearly_returns(nav_frame: pd.DataFrame) -> pd.DataFrame:
    monthly = build_monthly_returns(nav_frame)
    if monthly.empty:
        return pd.DataFrame(columns=["year", "return_pct"])
    yearly = (
        monthly.groupby("year")["monthly_return_pct"]
        .apply(lambda values: ((1 + values / 100).prod() - 1) * 100)
        .reset_index(name="return_pct")
    )
    return yearly.sort_values("year").reset_index(drop=True)


def build_calendar_year_comparison(
    fund_nav: pd.DataFrame,
    category_nav: pd.DataFrame,
) -> pd.DataFrame:
    fund_yearly = build_yearly_returns(fund_nav).rename(columns={"return_pct": "absolute_pct"})
    category_yearly = build_yearly_returns(category_nav).rename(columns={"return_pct": "category_average_pct"})
    comparison = fund_yearly.merge(category_yearly, on="year", how="left")
    comparison["period"] = comparison["year"].astype(str)
    return comparison[["period", "absolute_pct", "category_average_pct"]].sort_values("period", ascending=False).reset_index(drop=True)


@dataclass
class DrawdownEpisode:
    loss_pct: float
    start_date: pd.Timestamp
    start_value: float
    end_date: pd.Timestamp
    end_value: float
    length_days: int
    days_to_recover: int | None
    recovery_date: pd.Timestamp | None
    recovered_fraction: float


def calculate_drawdown_table(nav_frame: pd.DataFrame) -> pd.DataFrame:
    if nav_frame.empty:
        return pd.DataFrame(
            columns=[
                "loss_pct",
                "start_date",
                "start_value",
                "end_date",
                "end_value",
                "length_days",
                "days_to_recover",
                "recovery_date",
                "recovered_fraction",
            ]
        )

    frame = nav_frame.copy().sort_values("date").reset_index(drop=True)
    values = frame["value"].to_numpy(dtype=float)
    dates = frame["date"].to_numpy()

    episodes: list[DrawdownEpisode] = []
    peak_index = 0
    index = 1
    while index < len(frame):
        if values[index] >= values[peak_index]:
            peak_index = index
            index += 1
            continue

        valley_index = index
        while index + 1 < len(frame) and values[index + 1] < values[peak_index]:
            index += 1
            if values[index] < values[valley_index]:
                valley_index = index

        recovery_index = None
        probe = index + 1
        while probe < len(frame):
            if values[probe] >= values[peak_index]:
                recovery_index = probe
                break
            probe += 1

        recovered_fraction = values[-1] / values[peak_index] if values[peak_index] else 0.0
        if recovery_index is not None:
            recovered_fraction = 1.0

        episodes.append(
            DrawdownEpisode(
                loss_pct=((values[valley_index] / values[peak_index]) - 1) * 100,
                start_date=pd.Timestamp(dates[peak_index]),
                start_value=float(values[peak_index]),
                end_date=pd.Timestamp(dates[valley_index]),
                end_value=float(values[valley_index]),
                length_days=int((pd.Timestamp(dates[valley_index]) - pd.Timestamp(dates[peak_index])).days),
                days_to_recover=None if recovery_index is None else int((pd.Timestamp(dates[recovery_index]) - pd.Timestamp(dates[valley_index])).days),
                recovery_date=None if recovery_index is None else pd.Timestamp(dates[recovery_index]),
                recovered_fraction=float(recovered_fraction),
            )
        )

        if recovery_index is not None:
            peak_index = recovery_index
            index = recovery_index + 1
        else:
            break

    table = pd.DataFrame([episode.__dict__ for episode in episodes])
    if table.empty:
        return table
    return table.sort_values("loss_pct").head(10).reset_index(drop=True)


def build_drawdown_series(nav_frame: pd.DataFrame) -> pd.DataFrame:
    if nav_frame.empty:
        return pd.DataFrame(columns=["date", "drawdown_pct"])
    frame = nav_frame.copy().sort_values("date")
    frame["running_peak"] = frame["value"].cummax()
    frame["drawdown_pct"] = ((frame["value"] / frame["running_peak"]) - 1) * 100
    return frame[["date", "drawdown_pct"]]


def build_rolling_summary(rolling_returns: pd.DataFrame) -> dict[str, float | None]:
    if rolling_returns.empty:
        return {
            "mean_cagr": None,
            "median_cagr": None,
            "best_cagr": None,
            "worst_cagr": None,
        }
    return {
        "mean_cagr": float(rolling_returns["cagr_val"].mean()),
        "median_cagr": float(rolling_returns["cagr_val"].median()),
        "best_cagr": float(rolling_returns["cagr_val"].max()),
        "worst_cagr": float(rolling_returns["cagr_val"].min()),
    }


def compute_downside_rms(
    nav_frame: pd.DataFrame,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> dict[str, Any]:
    if nav_frame.empty:
        return {
            "rms_pct": None,
            "negative_day_pct": None,
            "positive_day_pct": None,
            "flat_day_pct": None,
            "daily_returns": pd.DataFrame(columns=["date", "daily_return_pct", "downside_component"]),
        }

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    frame = nav_frame.loc[(nav_frame["date"] >= start_ts) & (nav_frame["date"] <= end_ts)].copy().sort_values("date")
    if len(frame) < 2:
        return {
            "rms_pct": None,
            "negative_day_pct": None,
            "positive_day_pct": None,
            "flat_day_pct": None,
            "daily_returns": pd.DataFrame(columns=["date", "daily_return_pct", "downside_component"]),
        }

    frame["daily_return_pct"] = frame["value"].pct_change() * 100
    frame = frame.dropna(subset=["daily_return_pct"]).reset_index(drop=True)
    if frame.empty:
        return {
            "rms_pct": None,
            "negative_day_pct": None,
            "positive_day_pct": None,
            "flat_day_pct": None,
            "daily_returns": pd.DataFrame(columns=["date", "daily_return_pct", "downside_component"]),
        }

    frame["downside_component"] = frame["daily_return_pct"].clip(upper=0)
    frame["downside_component_sq"] = frame["downside_component"] ** 2
    rms_pct = float(np.sqrt(frame["downside_component_sq"].mean()))
    total_days = len(frame)
    negative_days = int((frame["daily_return_pct"] < 0).sum())
    positive_days = int((frame["daily_return_pct"] > 0).sum())
    flat_days = total_days - negative_days - positive_days

    return {
        "rms_pct": rms_pct,
        "negative_day_pct": negative_days / total_days * 100 if total_days else None,
        "positive_day_pct": positive_days / total_days * 100 if total_days else None,
        "flat_day_pct": flat_days / total_days * 100 if total_days else None,
        "daily_returns": frame[["date", "daily_return_pct", "downside_component"]],
    }


def build_fundamentals_table(fundamentals: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for key, value in fundamentals.items():
        if key.startswith("cat_avg_"):
            continue
        category_key = f"cat_avg_{key}"
        rows.append(
            {
                "metric": key.replace("_", " ").title(),
                "fund_value": safe_float(value),
                "category_average": safe_float(fundamentals.get(category_key)),
            }
        )
    return pd.DataFrame(rows)


def build_risk_table(risk_payload: dict[str, Any], horizon_key: str) -> pd.DataFrame:
    metric_order = ["returns", "risk_std_dev", "sharpe_ratio", "sortino_ratio", "beta"]
    labels = {
        "returns": "Returns",
        "risk_std_dev": "Standard Deviation",
        "sharpe_ratio": "Sharpe Ratio",
        "sortino_ratio": "Sortino Ratio",
        "beta": "Beta",
    }
    rows = []
    for metric_name in metric_order:
        metric_payload = risk_payload.get(metric_name, {})
        rows.append(
            {
                "metric": labels[metric_name],
                "fund_value": safe_float(metric_payload.get(horizon_key)),
                "category_average": safe_float(metric_payload.get(f"cat_avg_{horizon_key}")),
                "category_min": safe_float(metric_payload.get(f"cat_min_{horizon_key}")),
                "category_max": safe_float(metric_payload.get(f"cat_max_{horizon_key}")),
                "comment": build_risk_comment(metric_name, metric_payload, horizon_key),
            }
        )
    return pd.DataFrame(rows)


def build_risk_comment(metric_name: str, metric_payload: dict[str, Any], horizon_key: str) -> str:
    label = metric_payload.get(f"{horizon_key}_color", {}).get("str")
    value = safe_float(metric_payload.get(horizon_key))
    category_average = safe_float(metric_payload.get(f"cat_avg_{horizon_key}"))
    if value is None:
        return "Data unavailable for this horizon."
    if metric_name == "beta":
        return f"{label or 'Observed'}; beta closer to 1 usually means benchmark-like behaviour."
    if metric_name == "risk_std_dev":
        if category_average is None:
            return f"{label or 'Observed'}; lower standard deviation means smoother return swings."
        relation = "below" if value < category_average else "above"
        return f"{label or 'Observed'}; volatility is {relation} category average."
    if category_average is None:
        return f"{label or 'Observed'} for this period."
    relation = "above" if value > category_average else "below"
    return f"{label or 'Observed'}; this metric is {relation} category average."


def _scenario_slice(nav_frame: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    if nav_frame.empty:
        return pd.DataFrame(columns=["date", "value"]).copy()

    frame = nav_frame.copy()
    if "date" not in frame.columns:
        return pd.DataFrame(columns=["date", "value"]).copy()

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).copy()
    frame = frame.loc[(frame["date"] >= start_date) & (frame["date"] <= end_date)].copy()
    return frame.sort_values("date").reset_index(drop=True)


def _scenario_return(nav_frame: pd.DataFrame) -> float | None:
    if nav_frame.empty:
        return None
    start_value = float(nav_frame["value"].iloc[0])
    end_value = float(nav_frame["value"].iloc[-1])
    if start_value == 0:
        return None
    return ((end_value / start_value) - 1) * 100


def _scenario_volatility(nav_frame: pd.DataFrame) -> float | None:
    if len(nav_frame) < 2:
        return None
    daily_returns = nav_frame["value"].pct_change().dropna()
    if daily_returns.empty:
        return None
    return float(daily_returns.std() * np.sqrt(252) * 100)


def compute_scenario_metrics(
    fund_nav: pd.DataFrame,
    benchmark_nav: pd.DataFrame,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> dict[str, Any]:
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    fund_slice = _scenario_slice(fund_nav, start_ts, end_ts)
    benchmark_slice = _scenario_slice(benchmark_nav, start_ts, end_ts)

    if fund_slice.empty or benchmark_slice.empty:
        return {
            "start_date": start_ts,
            "end_date": end_ts,
            "fund_return_pct": None,
            "benchmark_return_pct": None,
            "excess_return_pct": None,
            "correlation": None,
            "fund_volatility_pct": None,
            "benchmark_volatility_pct": None,
            "plot_frame": pd.DataFrame(columns=["date", "value_fund", "value_benchmark"]),
        }

    fund_slice["date"] = pd.to_datetime(fund_slice["date"], errors="coerce")
    benchmark_slice["date"] = pd.to_datetime(benchmark_slice["date"], errors="coerce")
    fund_slice = fund_slice.dropna(subset=["date"]).copy()
    benchmark_slice = benchmark_slice.dropna(subset=["date"]).copy()

    if fund_slice.empty or benchmark_slice.empty:
        return {
            "start_date": start_ts,
            "end_date": end_ts,
            "fund_return_pct": None,
            "benchmark_return_pct": None,
            "excess_return_pct": None,
            "correlation": None,
            "fund_volatility_pct": None,
            "benchmark_volatility_pct": None,
            "plot_frame": pd.DataFrame(columns=["date", "value_fund", "value_benchmark"]),
        }

    merged = fund_slice.merge(benchmark_slice, on="date", how="inner", suffixes=("_fund", "_benchmark"))
    if merged.empty:
        merged = pd.DataFrame(columns=["date", "value_fund", "value_benchmark"])
    correlation = None
    if len(merged) >= 2:
        fund_returns = merged["value_fund"].pct_change().dropna()
        benchmark_returns = merged["value_benchmark"].pct_change().dropna()
        if len(fund_returns) >= 2 and len(benchmark_returns) >= 2:
            correlation = float(fund_returns.corr(benchmark_returns))
    fund_return = _scenario_return(fund_slice)
    benchmark_return = _scenario_return(benchmark_slice)
    return {
        "start_date": start_ts,
        "end_date": end_ts,
        "fund_return_pct": fund_return,
        "benchmark_return_pct": benchmark_return,
        "excess_return_pct": None if fund_return is None or benchmark_return is None else fund_return - benchmark_return,
        "correlation": correlation,
        "fund_volatility_pct": _scenario_volatility(fund_slice),
        "benchmark_volatility_pct": _scenario_volatility(benchmark_slice),
        "plot_frame": merged,
    }


def calculate_diversification_snapshot(portfolio_payload: dict[str, Any]) -> dict[str, float | None]:
    summary = portfolio_payload.get("summary", {})
    concentration = summary.get("concentration", {})
    return {
        "number_of_holdings": safe_float(concentration.get("number_of_holding")),
        "top_10_weight_pct": safe_float(concentration.get("top_10_stk_wt")),
        "top_5_weight_pct": safe_float(concentration.get("top_5_stk_wt")),
        "top_3_sector_weight_pct": safe_float(concentration.get("top_3_sector_wt")),
    }
