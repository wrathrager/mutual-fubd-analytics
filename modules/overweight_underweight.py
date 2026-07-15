from __future__ import annotations

from typing import Any
import difflib

import pandas as pd


def compute_overweight_score(
    fund_weights: pd.DataFrame,
    sector_returns: pd.DataFrame,
    sector_mapping: dict[str, list[str]],
    price_history: pd.DataFrame,
    lookback_label: str,
) -> dict[str, Any]:
    timeline_returns = sector_returns.loc[sector_returns["Lookback"] == lookback_label].copy()
    if timeline_returns.empty:
        raise ValueError(f"No sector returns found for {lookback_label}.")

    top_five = timeline_returns.sort_values("Return", ascending=False).head(5).copy()
    top_five["bucket"] = "Top 5"
    bottom_five = timeline_returns.sort_values("Return", ascending=True).head(5).copy()
    bottom_five["bucket"] = "Bottom 5"
    focus_sectors = pd.concat([top_five, bottom_five], ignore_index=True)

    return_start = pd.Timestamp(focus_sectors["TargetDateUsed"].iloc[0])
    return_end = pd.Timestamp(focus_sectors["AnchorDateUsed"].iloc[0])
    monthly_subwindows = build_monthly_subwindows(return_start=return_start, return_end=return_end)

    available_dates = sorted(fund_weights["date"].dropna().unique())
    summary_signal_start_date = None
    summary_signal_end_date = None

    resolved_mapping, mapping_issues = resolve_sector_mapping(
        sector_mapping=sector_mapping,
        available_sectors=fund_weights["sector"].dropna().unique().tolist(),
    )

    sector_rows: list[dict[str, Any]] = []
    all_funds = sorted(fund_weights["fund_name"].dropna().unique().tolist())

    for _, sector_row in focus_sectors.iterrows():
        index_name = sector_row["Stock"]
        mapped_sectors = resolved_mapping.get(index_name, [])
        if not mapped_sectors:
            mapping_issues.append(f"No weight-sector mapping resolved for {index_name}")
            continue

        mapped_history = (
            fund_weights.loc[fund_weights["sector"].isin(mapped_sectors)]
            .groupby(["fund_name", "date"], as_index=False)["weight_pct"]
            .sum()
        )
        lookup = {
            (row.fund_name, row.date): float(row.weight_pct)
            for row in mapped_history.itertuples(index=False)
        }
        monthly_returns = get_monthly_sector_returns(
            price_history=price_history,
            index_name=index_name,
            monthly_subwindows=monthly_subwindows,
        )
        signal_start_date, signal_end_date = determine_signal_window(
            available_dates=available_dates,
            return_start=return_start,
        )
        if should_shift_to_late_window(
            lookback_label=lookback_label,
            monthly_returns=monthly_returns,
            sector_return_pct=float(sector_row["Return"]),
        ):
            late_window_start = monthly_returns[-1]["subwindow_start"]
            signal_start_date, signal_end_date = determine_signal_window(
                available_dates=available_dates,
                return_start=late_window_start,
            )

        if summary_signal_start_date is None:
            summary_signal_start_date = signal_start_date
            summary_signal_end_date = signal_end_date

        sector_detail_rows: list[dict[str, Any]] = []
        for fund_name in all_funds:
            start_weight = lookup.get((fund_name, signal_start_date), 0.0)
            end_weight = lookup.get((fund_name, signal_end_date), 0.0)
            delta_weight = end_weight - start_weight
            latest_aligned_delta = delta_weight if sector_row["Return"] >= 0 else -delta_weight

            rolling_aligned_components: list[float] = []
            rolling_weighted_sum = 0.0
            rolling_signal_pairs: list[str] = []
            for monthly_return_row in monthly_returns:
                monthly_signal_start, monthly_signal_end = determine_signal_window(
                    available_dates=available_dates,
                    return_start=monthly_return_row["subwindow_start"],
                )
                monthly_start_weight = lookup.get((fund_name, monthly_signal_start), 0.0)
                monthly_end_weight = lookup.get((fund_name, monthly_signal_end), 0.0)
                monthly_delta = monthly_end_weight - monthly_start_weight
                monthly_aligned_delta = (
                    monthly_delta
                    if monthly_return_row["monthly_return_pct"] >= 0
                    else -monthly_delta
                )
                rolling_aligned_components.append(monthly_aligned_delta)
                rolling_weighted_sum += monthly_aligned_delta * abs(monthly_return_row["monthly_return_pct"])
                rolling_signal_pairs.append(
                    f"{monthly_signal_start.date()}->{monthly_signal_end.date()} for {monthly_return_row['subwindow_start'].date()}->{monthly_return_row['subwindow_end'].date()}"
                )

            sector_detail_rows.append(
                {
                    "fund_name": fund_name,
                    "index_name": index_name,
                    "bucket": sector_row["bucket"],
                    "sector_return_pct": float(sector_row["Return"]),
                    "mapped_weight_sectors": ", ".join(mapped_sectors),
                    "signal_start_date": signal_start_date,
                    "signal_end_date": signal_end_date,
                    "start_weight_pct": start_weight,
                    "end_weight_pct": end_weight,
                    "delta_weight_pct": delta_weight,
                    "aligned_delta_pct": latest_aligned_delta,
                    "latest_aligned_delta_pct": latest_aligned_delta,
                    "rolling_aligned_delta_pct": float(sum(rolling_aligned_components)),
                    "rolling_weighted_alignment": rolling_weighted_sum,
                    "rolling_signal_windows": " | ".join(rolling_signal_pairs),
                    "favorable_move": latest_aligned_delta > 0,
                }
            )

        sector_detail = pd.DataFrame(sector_detail_rows)
        latest_abs = sector_detail["latest_aligned_delta_pct"].abs().max()
        latest_normalizer = latest_abs if pd.notna(latest_abs) and latest_abs > 0 else 1.0
        rolling_abs = sector_detail["rolling_weighted_alignment"].abs().max()
        rolling_normalizer = rolling_abs if pd.notna(rolling_abs) and rolling_abs > 0 else 1.0

        sector_detail["normalized_latest_alignment"] = sector_detail["latest_aligned_delta_pct"] / latest_normalizer
        sector_detail["normalized_rolling_alignment"] = sector_detail["rolling_weighted_alignment"] / rolling_normalizer
        sector_detail["latest_sector_score"] = (
            sector_detail["normalized_latest_alignment"] * abs(float(sector_row["Return"]))
        )
        sector_detail["rolling_sector_score"] = (
            sector_detail["normalized_rolling_alignment"] * abs(float(sector_row["Return"]))
        )
        sector_detail["sector_score"] = (
            0.5 * sector_detail["latest_sector_score"] + 0.5 * sector_detail["rolling_sector_score"]
        )
        sector_rows.extend(sector_detail.to_dict("records"))

    detail_df = pd.DataFrame(sector_rows)
    if detail_df.empty:
        raise ValueError("No sector rows could be scored after applying the mappings.")

    rankings = build_rankings(detail_df)
    return {
        "module_name": "overweight_underweight",
        "score_column": "total_score",
        "rankings": rankings,
        "sector_details": detail_df.sort_values(
            ["index_name", "sector_score"],
            ascending=[True, False],
        ).reset_index(drop=True),
        "focus_sectors": focus_sectors.reset_index(drop=True),
        "summary": {
            "lookback_label": lookback_label,
            "return_start": return_start,
            "return_end": return_end,
            "signal_start_date": signal_start_date,
            "signal_end_date": signal_end_date,
            "fund_count": len(all_funds),
            "sector_count": int(focus_sectors["Stock"].nunique()),
        },
        "notes": [
            f"Return ranking dates come directly from stock_performance_ranks1.csv: {return_start.date()} to {return_end.date()}.",
            (
                f"Pre-positioning is measured using fund disclosures from {summary_signal_start_date.date()} "
                f"to {summary_signal_end_date.date()}, using the two disclosures immediately before the relevant return window."
            ),
            "A fund is rewarded when it increased mapped sector weights before a positive sector move or reduced them before a negative move.",
            "Per-sector scoring now blends two views: the latest pre-window signal and a rolling pre-positioning signal across the monthly subwindows inside the selected return horizon.",
            "Top 5 / Bottom 5 sector selection still comes from stock_performance_ranks1.csv, but the rolling pre-positioning component now uses raw daily index prices from PRICES.xlsx to calculate the internal monthly subwindow moves.",
            "The rolling component helps credit managers who built positions earlier during a flatter phase before a late spike or late fall.",
            "For longer lookbacks, the latest pre-positioning signal shifts to the disclosures immediately before the last monthly subwindow when that subwindow dominates the overall return profile.",
            "The current blend is 50% latest pre-window signal and 50% rolling pre-positioning signal.",
        ],
        "issues": sorted(set(mapping_issues)),
    }


def determine_signal_window(
    available_dates: list[pd.Timestamp],
    return_start: pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    parsed_dates = sorted(pd.Timestamp(value) for value in available_dates)
    prior_dates = [date for date in parsed_dates if date < return_start]
    if not prior_dates:
        raise ValueError(
            f"Need at least one weight disclosure date before {return_start.date()}, found 0."
        )

    signal_end = prior_dates[-1]
    signal_start = prior_dates[-2] if len(prior_dates) >= 2 else signal_end
    return signal_start, signal_end


def should_shift_to_late_window(
    lookback_label: str,
    monthly_returns: list[dict[str, Any]],
    sector_return_pct: float,
) -> bool:
    if lookback_label == "1M" or len(monthly_returns) < 2:
        return False

    total_abs_return = sum(abs(row["monthly_return_pct"]) for row in monthly_returns)
    if total_abs_return <= 0:
        return False

    last_return = monthly_returns[-1]["monthly_return_pct"]
    if abs(last_return) / total_abs_return < 0.5:
        return False

    return (last_return >= 0 and sector_return_pct >= 0) or (last_return < 0 and sector_return_pct < 0)


def build_monthly_subwindows(
    return_start: pd.Timestamp,
    return_end: pd.Timestamp,
) -> list[dict[str, pd.Timestamp]]:
    subwindows: list[dict[str, pd.Timestamp]] = []
    current_start = pd.Timestamp(return_start)
    while current_start < return_end:
        current_end = min(current_start + pd.DateOffset(months=1), return_end)
        subwindows.append(
            {
                "subwindow_start": current_start,
                "subwindow_end": current_end,
            }
        )
        current_start = current_end
    return subwindows


def get_monthly_sector_returns(
    price_history: pd.DataFrame,
    index_name: str,
    monthly_subwindows: list[dict[str, pd.Timestamp]],
) -> list[dict[str, Any]]:
    monthly_rows: list[dict[str, Any]] = []
    if index_name not in price_history.columns:
        return monthly_rows

    index_prices = price_history[["date", index_name]].copy()
    index_prices[index_name] = pd.to_numeric(index_prices[index_name], errors="coerce")
    index_prices = index_prices.dropna(subset=[index_name]).sort_values("date").reset_index(drop=True)
    if index_prices.empty:
        return monthly_rows

    for subwindow in monthly_subwindows:
        start_value = find_price_at_or_after(index_prices, index_name, subwindow["subwindow_start"])
        end_value = find_price_at_or_before(index_prices, index_name, subwindow["subwindow_end"])
        if start_value is None or end_value is None:
            continue
        start_date, start_price = start_value
        end_date, end_price = end_value
        if start_price == 0:
            continue
        monthly_rows.append(
            {
                "subwindow_start": subwindow["subwindow_start"],
                "subwindow_end": subwindow["subwindow_end"],
                "effective_start_date": start_date,
                "effective_end_date": end_date,
                "monthly_return_pct": ((end_price / start_price) - 1) * 100,
            }
        )
    return monthly_rows


def find_price_at_or_after(
    price_frame: pd.DataFrame,
    column: str,
    target_date: pd.Timestamp,
) -> tuple[pd.Timestamp, float] | None:
    eligible = price_frame.loc[price_frame["date"] >= target_date].sort_values("date")
    if eligible.empty:
        return None
    row = eligible.iloc[0]
    return pd.Timestamp(row["date"]), float(row[column])


def find_price_at_or_before(
    price_frame: pd.DataFrame,
    column: str,
    target_date: pd.Timestamp,
) -> tuple[pd.Timestamp, float] | None:
    eligible = price_frame.loc[price_frame["date"] <= target_date].sort_values("date")
    if eligible.empty:
        return None
    row = eligible.iloc[-1]
    return pd.Timestamp(row["date"]), float(row[column])


def resolve_sector_mapping(
    sector_mapping: dict[str, list[str]],
    available_sectors: list[str],
) -> tuple[dict[str, list[str]], list[str]]:
    available_lookup = {normalize_name(sector): sector for sector in available_sectors}
    resolved: dict[str, list[str]] = {}
    issues: list[str] = []

    for index_name, mapped_sectors in sector_mapping.items():
        resolved_sectors: list[str] = []
        for sector in mapped_sectors:
            normalized = normalize_name(sector)
            exact = available_lookup.get(normalized)
            if exact:
                resolved_sectors.append(exact)
                continue

            close_match = difflib.get_close_matches(
                normalized,
                list(available_lookup.keys()),
                n=1,
                cutoff=0.82,
            )
            if close_match:
                resolved_sectors.append(available_lookup[close_match[0]])
            else:
                issues.append(f"Unmatched mapped sector '{sector}' for {index_name}")

        resolved[index_name] = sorted(set(resolved_sectors))

    return resolved, issues


def normalize_name(value: str) -> str:
    return " ".join(str(value).replace("&", "and").replace("/", " ").replace("-", " ").split()).lower()


def build_rankings(detail_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        detail_df.groupby("fund_name", as_index=False)
        .agg(
            total_score=("sector_score", "sum"),
            favorable_hits=("favorable_move", "sum"),
            unfavorable_hits=("favorable_move", lambda values: int((~values).sum())),
            avg_aligned_delta_pct=("aligned_delta_pct", "mean"),
            best_sector_score=("sector_score", "max"),
            worst_sector_score=("sector_score", "min"),
        )
        .sort_values(["total_score", "favorable_hits"], ascending=[False, False])
        .reset_index(drop=True)
    )

    best_sector = (
        detail_df.sort_values("sector_score", ascending=False)
        .groupby("fund_name", as_index=False)
        .first()[["fund_name", "index_name", "sector_score"]]
        .rename(columns={"index_name": "best_sector", "sector_score": "best_sector_score_detail"})
    )
    worst_sector = (
        detail_df.sort_values("sector_score", ascending=True)
        .groupby("fund_name", as_index=False)
        .first()[["fund_name", "index_name", "sector_score"]]
        .rename(columns={"index_name": "worst_sector", "sector_score": "worst_sector_score_detail"})
    )
    return summary.merge(best_sector, on="fund_name", how="left").merge(
        worst_sector,
        on="fund_name",
        how="left",
    )
