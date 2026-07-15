from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def normalize_series(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index, dtype=float)
    min_value = float(valid.min())
    max_value = float(valid.max())
    if np.isclose(min_value, max_value):
        normalized = pd.Series(100.0, index=series.index, dtype=float)
    else:
        normalized = (numeric - min_value) / (max_value - min_value) * 100
    if not higher_is_better:
        normalized = 100 - normalized
    return normalized


def build_returns_scores(snapshot: pd.DataFrame) -> pd.DataFrame:
    frame = snapshot[["fund_name", "coarse_fund_key", "return_1y_pct", "return_3y_pct", "return_5y_pct"]].copy()
    frame["returns_1y_score"] = normalize_series(frame["return_1y_pct"], higher_is_better=True)
    frame["returns_3y_score"] = normalize_series(frame["return_3y_pct"], higher_is_better=True)
    frame["returns_5y_score"] = normalize_series(frame["return_5y_pct"], higher_is_better=True)
    return frame[["fund_name", "coarse_fund_key", "returns_1y_score", "returns_3y_score", "returns_5y_score"]]


def build_rolling_scores(snapshot: pd.DataFrame) -> pd.DataFrame:
    frame = snapshot[["fund_name", "coarse_fund_key", "rolling_1y_median_cagr", "rolling_3y_median_cagr", "rolling_5y_median_cagr"]].copy()
    frame["rollingreturns_1y_score"] = normalize_series(frame["rolling_1y_median_cagr"], higher_is_better=True)
    frame["rollingreturns_3y_score"] = normalize_series(frame["rolling_3y_median_cagr"], higher_is_better=True)
    frame["rollingreturns_5y_score"] = normalize_series(frame["rolling_5y_median_cagr"], higher_is_better=True)
    return frame[["fund_name", "coarse_fund_key", "rollingreturns_1y_score", "rollingreturns_3y_score", "rollingreturns_5y_score"]]


def build_diversification_scores(snapshot: pd.DataFrame) -> pd.DataFrame:
    frame = snapshot[
        ["fund_name", "coarse_fund_key", "number_of_holdings", "top_10_weight_pct", "top_5_weight_pct", "top_3_sector_weight_pct"]
    ].copy()
    frame["holdings_component"] = normalize_series(frame["number_of_holdings"], higher_is_better=True)
    frame["top10_component"] = normalize_series(frame["top_10_weight_pct"], higher_is_better=False)
    frame["top5_component"] = normalize_series(frame["top_5_weight_pct"], higher_is_better=False)
    frame["top3sector_component"] = normalize_series(frame["top_3_sector_weight_pct"], higher_is_better=False)
    frame["diversification_score"] = frame[
        ["holdings_component", "top10_component", "top5_component", "top3sector_component"]
    ].mean(axis=1, skipna=True)
    return frame[["fund_name", "coarse_fund_key", "diversification_score"]]


def build_fundamentals_scores(snapshot: pd.DataFrame) -> pd.DataFrame:
    frame = snapshot[
        ["fund_name", "coarse_fund_key", "pe", "pb", "price_sale", "price_cash_flow", "dividend_yield", "roe"]
    ].copy()
    frame["pe_component"] = normalize_series(frame["pe"], higher_is_better=False)
    frame["pb_component"] = normalize_series(frame["pb"], higher_is_better=False)
    frame["ps_component"] = normalize_series(frame["price_sale"], higher_is_better=False)
    frame["pcf_component"] = normalize_series(frame["price_cash_flow"], higher_is_better=False)
    frame["dividend_component"] = normalize_series(frame["dividend_yield"], higher_is_better=True)
    frame["roe_component"] = normalize_series(frame["roe"], higher_is_better=True)
    frame["fundamentals_score"] = frame[
        [
            "pe_component",
            "pb_component",
            "ps_component",
            "pcf_component",
            "dividend_component",
            "roe_component",
        ]
    ].mean(axis=1, skipna=True)
    return frame[["fund_name", "coarse_fund_key", "fundamentals_score"]]


def build_risk_scores(snapshot: pd.DataFrame) -> pd.DataFrame:
    frame = snapshot[
        ["fund_name", "coarse_fund_key", "std_dev_1y", "std_dev_3y", "std_dev_5y", "sharpe_1y", "sharpe_3y", "sharpe_5y", "sortino_1y", "sortino_3y", "sortino_5y", "beta_1y", "beta_3y", "beta_5y"]
    ].copy()
    for horizon in ["1y", "3y", "5y"]:
        frame[f"std_component_{horizon}"] = normalize_series(frame[f"std_dev_{horizon}"], higher_is_better=False)
        frame[f"sharpe_component_{horizon}"] = normalize_series(frame[f"sharpe_{horizon}"], higher_is_better=True)
        frame[f"sortino_component_{horizon}"] = normalize_series(frame[f"sortino_{horizon}"], higher_is_better=True)
        frame[f"beta_component_{horizon}"] = normalize_series((frame[f"beta_{horizon}"] - 1).abs(), higher_is_better=False)
    component_columns = [column for column in frame.columns if column.endswith("_component_1y") or column.endswith("_component_3y") or column.endswith("_component_5y")]
    frame["risk_metrics_score"] = frame[component_columns].mean(axis=1, skipna=True)
    return frame[["fund_name", "coarse_fund_key", "risk_metrics_score"]]


def build_drawdown_scores(snapshot: pd.DataFrame) -> pd.DataFrame:
    frame = snapshot[["fund_name", "coarse_fund_key", "max_drawdown_pct", "avg_drawdown_recovery_days"]].copy()
    frame["depth_component"] = normalize_series(frame["max_drawdown_pct"], higher_is_better=True)
    frame["recovery_component"] = normalize_series(frame["avg_drawdown_recovery_days"], higher_is_better=False)
    frame["drawdown_score"] = frame[["depth_component", "recovery_component"]].mean(axis=1, skipna=True)
    return frame[["fund_name", "coarse_fund_key", "drawdown_score"]]


def build_stress_scores(snapshot: pd.DataFrame) -> pd.DataFrame:
    frame = snapshot[["fund_name", "coarse_fund_key", "stress_avg_excess_return", "stress_avg_correlation"]].copy()
    frame["excess_component"] = normalize_series(frame["stress_avg_excess_return"], higher_is_better=True)
    frame["correlation_component"] = normalize_series(frame["stress_avg_correlation"], higher_is_better=False)
    frame["stress_test_score"] = frame[["excess_component", "correlation_component"]].mean(axis=1, skipna=True)
    return frame[["fund_name", "coarse_fund_key", "stress_test_score"]]


def build_rms_scores(snapshot: pd.DataFrame) -> pd.DataFrame:
    frame = snapshot[["fund_name", "coarse_fund_key", "downside_rms_pct"]].copy()
    frame["rms_downside_score"] = normalize_series(frame["downside_rms_pct"], higher_is_better=False)
    return frame[["fund_name", "coarse_fund_key", "rms_downside_score"]]


def merge_score_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    merged = frames[0].copy()
    for frame in frames[1:]:
        join_columns = ["coarse_fund_key"] if "coarse_fund_key" in merged.columns and "coarse_fund_key" in frame.columns else ["fund_name"]
        merged = merged.merge(frame, on=join_columns, how="outer", suffixes=("", "_dup"))
        if "fund_name_dup" in merged.columns:
            merged["fund_name"] = merged["fund_name"].fillna(merged["fund_name_dup"])
            merged = merged.drop(columns=["fund_name_dup"])
    score_columns = [column for column in merged.columns if column not in {"fund_name", "coarse_fund_key"}]
    if "coarse_fund_key" in merged.columns:
        aggregated = (
            merged.groupby("coarse_fund_key", as_index=False)[score_columns]
            .mean(numeric_only=True)
        )
        display_names = (
            merged.groupby("coarse_fund_key")["fund_name"]
            .agg(lambda values: sorted([value for value in values.dropna().unique()], key=len, reverse=True)[0] if len(values.dropna()) else None)
            .reset_index()
        )
        merged = display_names.merge(aggregated, on="coarse_fund_key", how="left")
    return merged


def build_final_ranker(
    raw_scores: pd.DataFrame,
    weight_map: dict[str, float],
) -> pd.DataFrame:
    score_columns = [column for column, weight in weight_map.items() if column in raw_scores.columns and weight > 0]
    if not score_columns:
        return pd.DataFrame()
    total = sum(weight_map[column] for column in score_columns)
    if total <= 0:
        return pd.DataFrame()
    weights = {column: weight_map[column] / total for column in score_columns}

    frame = raw_scores.copy()
    weighted_parts = []
    active_weight_sum = pd.Series(0.0, index=frame.index)
    for column, weight in weights.items():
        valid = frame[column].notna()
        active_weight_sum = active_weight_sum + valid.astype(float) * weight
        weighted_parts.append(frame[column].fillna(0) * weight)
    frame["final_score"] = sum(weighted_parts)
    frame["final_score"] = np.where(active_weight_sum > 0, frame["final_score"] / active_weight_sum, np.nan)
    frame = frame.sort_values("final_score", ascending=False).reset_index(drop=True)
    frame["rank"] = range(1, len(frame) + 1)
    return frame


def build_metric_methodology() -> dict[str, str]:
    return {
        "overweight_1m_score": "Normalized 1M overweight/underweight total_score from the sector rotation engine.",
        "overweight_3m_score": "Normalized 3M overweight/underweight total_score from the sector rotation engine.",
        "overweight_6m_score": "Normalized 6M overweight/underweight total_score from the sector rotation engine.",
        "returns_5y_score": "Normalized 5Y CAGR computed from fund NAV up to 31 May 2026.",
        "returns_3y_score": "Normalized 3Y CAGR computed from fund NAV up to 31 May 2026.",
        "returns_1y_score": "Normalized 1Y CAGR computed from fund NAV up to 31 May 2026.",
        "rollingreturns_5y_score": "Normalized median 5Y rolling CAGR from the rolling returns endpoint.",
        "rollingreturns_3y_score": "Normalized median 3Y rolling CAGR from the rolling returns endpoint.",
        "rollingreturns_1y_score": "Normalized median 1Y rolling CAGR from the rolling returns endpoint.",
        "diversification_score": "Average of normalized breadth and inverse concentration metrics from the portfolio snapshot.",
        "fundamentals_score": "Average of normalized valuation and quality ratios from the fundamentals snapshot.",
        "risk_metrics_score": "Average of normalized volatility, Sharpe, Sortino, and beta components from risk metrics.",
        "drawdown_score": "Average of normalized inverse drawdown depth and inverse recovery time from NAV history up to 31 May 2026.",
        "stress_test_score": "Average of normalized excess return and inverse correlation across built-in stress scenarios.",
        "rms_downside_score": "Normalized inverse downside RMS of daily returns over the selected analysis window; lower downside RMS is better.",
    }
