from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from aggregator import ScoreModule, run_score_modules
from modules import (
    SCORING_CUTOFF_DATE,
    coarse_fund_key,
    compute_overweight_score,
    fetch_fundamentals,
    fetch_nav_bundle,
    fetch_nifty100_series,
    fetch_portfolio,
    fetch_risk_metrics,
    fetch_rolling_returns,
    load_fund_master,
    normalize_fund_name,
)
from modules.fund_analytics import (
    BUILT_IN_SCENARIOS,
    build_drawdown_series,
    build_fundamentals_table,
    build_monthly_heatmap,
    build_monthly_returns,
    build_risk_table,
    build_rolling_summary,
    build_yearly_returns,
    calculate_cagr,
    calculate_diversification_snapshot,
    calculate_drawdown_table,
    compute_downside_rms,
    compute_scenario_metrics,
    filter_to_cutoff,
    safe_float,
)
from modules.fund_scores import (
    build_diversification_scores,
    build_drawdown_scores,
    build_final_ranker,
    build_fundamentals_scores,
    build_metric_methodology,
    build_rms_scores,
    build_returns_scores,
    build_risk_scores,
    build_rolling_scores,
    build_stress_scores,
    merge_score_frames,
    normalize_series,
)
from weights_calc import build_fund_sector_history


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RANKS_PATH = DATA_DIR / "stock_performance_ranks1.csv"
MAPPINGS_PATH = DATA_DIR / "mappings.xlsx"
RAW_WEIGHTS_DIR = DATA_DIR / "weight_change"
FUND_HISTORY_CACHE = DATA_DIR / "fund_sector_weights.csv"
ISIN_MASTER_PATH = DATA_DIR / "isin_largecap.xlsx"
PRICES_PATH = BASE_DIR / "PRICES.xlsx"

SECTION_OPTIONS = [
    "Home",
    "Overweight / Underweight",
    "Fund Analytics",
    "Comparative Insights",
    "Final Ranker",
]


def main() -> None:
    st.set_page_config(page_title="Mutual Fund Analytics Platform", layout="wide")
    st.title("Mutual Fund Analytics Platform")
    st.caption("Modular MVP with expandable sections for sector timing, fund analytics, comparisons, and final ranking.")

    datasets = load_core_datasets()

    with st.sidebar:
        st.header("Navigation")
        selected_section = st.radio("Go to section", options=SECTION_OPTIONS)

    if selected_section == "Home":
        render_home_page()
    elif selected_section == "Overweight / Underweight":
        render_overweight_page(datasets)
    elif selected_section == "Fund Analytics":
        render_fund_analytics_page(datasets)
    elif selected_section == "Comparative Insights":
        render_comparative_page(datasets)
    elif selected_section == "Final Ranker":
        render_final_ranker_page(datasets)


@st.cache_data(show_spinner=False)
def load_core_datasets() -> dict[str, object]:
    sector_returns = pd.read_csv(RANKS_PATH)
    for column in ["AnchorDateUsed", "TargetDateUsed"]:
        sector_returns[column] = pd.to_datetime(
            sector_returns[column],
            format="%d-%m-%Y",
            errors="coerce",
        )
    sector_returns["Lookback"] = sector_returns["Lookback"].astype(str).str.strip()
    sector_returns["Stock"] = sector_returns["Stock"].astype(str).str.strip()

    mapping_frame = pd.read_excel(MAPPINGS_PATH)
    mapping_frame.columns = ["index_name", "weight_sector"]
    mapping_frame["index_name"] = mapping_frame["index_name"].ffill()
    mapping_frame["index_name"] = mapping_frame["index_name"].astype(str).str.strip()
    mapping_frame["weight_sector"] = mapping_frame["weight_sector"].astype(str).str.strip()
    sector_mapping = (
        mapping_frame.groupby("index_name")["weight_sector"]
        .apply(lambda values: [value for value in values if value and value.lower() != "nan"])
        .to_dict()
    )

    fund_weights = build_fund_sector_history(
        raw_dir=RAW_WEIGHTS_DIR,
        output_path=FUND_HISTORY_CACHE,
    )
    fund_weights["date"] = pd.to_datetime(fund_weights["date"])
    fund_weights["fund_name"] = fund_weights["fund_name"].astype(str).str.strip()
    fund_weights["sector"] = fund_weights["sector"].astype(str).str.strip()
    fund_weights["fund_key"] = fund_weights["fund_name"].map(normalize_fund_name)

    fund_master = load_fund_master(str(ISIN_MASTER_PATH))
    price_frame = pd.read_excel(PRICES_PATH)
    price_frame = price_frame.rename(columns={"DATE": "date"}).copy()
    price_frame["date"] = pd.to_datetime(price_frame["date"], errors="coerce")
    price_frame = price_frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return {
        "sector_returns": sector_returns,
        "sector_mapping": sector_mapping,
        "fund_weights": fund_weights,
        "fund_master": fund_master,
        "price_history": price_frame,
    }


def render_home_page() -> None:
    st.subheader("Available sections")
    left, right = st.columns(2)
    with left:
        st.markdown("**Overweight / Underweight**")
        st.write("Sector rotation engine based on top/bottom sector moves and prior monthly portfolio weight changes.")
        st.markdown("**Fund Analytics**")
        st.write("Single-fund deep dive with NAV, heatmaps, rolling returns, holdings, fundamentals, risk, and drawdowns.")
    with right:
        st.markdown("**Comparative Insights**")
        st.write("Cross-fund bar comparisons and stress-test behaviour versus Nifty 100.")
        st.markdown("**Final Ranker**")
        st.write("User-prioritized composite ranking using normalized raw scores from all active modules.")

    with st.expander("How scoring consistency is handled", expanded=False):
        st.write(
            f"- Sector timing scores use the disclosed weight history available up to `2026-05-31`.\n"
            f"- NAV-based scores in later sections are also capped at `{SCORING_CUTOFF_DATE.date()}` for ranking consistency.\n"
            "- Current-snapshot APIs like holdings, fundamentals, and Moneycontrol risk metrics are shown fully in the UI; where they contribute to scores, they are labelled as current-snapshot proxies."
        )


def render_overweight_page(datasets: dict[str, object]) -> None:
    st.subheader("Overweight / Underweight")
    st.caption("Separate section for the sector rotation engine. Existing functionality stays intact and future modules can sit alongside it.")

    available_lookbacks = sorted(
        [
            label
            for label in datasets["sector_returns"]["Lookback"].unique().tolist()
            if parse_lookback_months(label) <= 9
        ],
        key=parse_lookback_months,
    )

    selected_lookbacks = st.multiselect(
        "Select one or more lookbacks",
        options=available_lookbacks,
        default=["1M", "3M"],
        format_func=lambda label: label.replace("M", " Month"),
        key="ow_lookbacks",
    )

    if not selected_lookbacks:
        st.info("Select at least one lookback from 1M to 9M.")
        return

    modules = [
        ScoreModule(
            name="overweight_underweight",
            weight=1.0,
            compute=compute_overweight_score,
        )
    ]

    all_issues: dict[str, str] = {}
    for lookback_label in selected_lookbacks:
        try:
            results = run_score_modules(
                modules=modules,
                fund_weights=datasets["fund_weights"],
                sector_returns=datasets["sector_returns"],
                sector_mapping=datasets["sector_mapping"],
                price_history=datasets["price_history"],
                lookback_label=lookback_label,
            )
            module_result = results["module_outputs"].get("overweight_underweight")
            if not module_result:
                raise ValueError(results["errors"].get("overweight_underweight", "No module output returned."))
            render_lookback_section(lookback_label, module_result)
            all_issues.update({f"{lookback_label}:{key}": value for key, value in results["errors"].items()})
        except Exception as exc:
            all_issues[lookback_label] = str(exc)
            st.error(f"{lookback_label} could not be computed: {exc}")

    render_run_status(all_issues)


def render_fund_analytics_page(datasets: dict[str, object]) -> None:
    st.subheader("Fund Analytics")
    st.caption("Single-fund analytics view using Moneycontrol NAV, rolling returns, holdings, fundamentals, and risk endpoints.")

    fund_master = datasets["fund_master"]
    selected_universe = st.multiselect(
        "Fund universe",
        options=fund_master["fund_name"].tolist(),
        default=fund_master["fund_name"].tolist(),
        key="fa_universe",
    )
    filtered_master = fund_master.loc[fund_master["fund_name"].isin(selected_universe)].copy()
    if filtered_master.empty:
        st.info("Select at least one fund in the universe.")
        return

    fund_name = st.selectbox("Select a fund", options=filtered_master["fund_name"].tolist(), key="fa_fund")
    fund_row = fund_master.loc[fund_master["fund_name"] == fund_name].iloc[0]

    try:
        max_nav_bundle = fetch_nav_bundle(fund_row["isin"])
    except Exception as exc:
        st.error(f"Could not fetch NAV data for {fund_name}: {exc}")
        return

    nav_options = max_nav_bundle["available_periods"] or [max_nav_bundle["duration"]]
    nav_index = nav_options.index(max_nav_bundle["duration"]) if max_nav_bundle["duration"] in nav_options else 0
    nav_duration = st.selectbox(
        "NAV duration",
        options=nav_options,
        index=nav_index,
        key="fa_nav_duration",
    )
    nav_bundle = fetch_nav_bundle(fund_row["isin"], nav_duration)
    fund_nav = nav_bundle["fund_nav"]
    scoring_nav = filter_to_cutoff(fund_nav)

    cagr_cols = st.columns(3)
    cagr_cols[0].metric("1Y CAGR", format_pct(calculate_cagr(fund_nav, 1)))
    cagr_cols[1].metric("3Y CAGR", format_pct(calculate_cagr(fund_nav, 3)))
    cagr_cols[2].metric("5Y CAGR", format_pct(calculate_cagr(fund_nav, 5)))

    with st.expander("How this section is calculated", expanded=False):
        st.write(
            "- NAV charts and heatmaps use the full live series fetched for the chosen duration.\n"
            f"- Historical ranking scores later in the app use NAV capped at `{SCORING_CUTOFF_DATE.date()}` to stay aligned with the May 2026 sector-weight dataset.\n"
            "- The NAV duration dropdown falls back safely if Moneycontrol omits the requested period label for a fund."
        )

    render_nav_chart(fund_nav, fund_name)

    analytics_tabs = st.tabs(["Return Heatmaps", "Rolling Returns", "Portfolio", "Fundamentals", "Risk & Drawdown"])

    with analytics_tabs[0]:
        render_return_heatmaps_and_bars(fund_nav)

    with analytics_tabs[1]:
        safe_render_section("Rolling Returns", render_rolling_returns_section, fund_row["isin"])

    with analytics_tabs[2]:
        safe_render_section("Portfolio", render_portfolio_section, fund_row["isin"])

    with analytics_tabs[3]:
        safe_render_section("Fundamentals", render_fundamentals_section, fund_row["isin"])

    with analytics_tabs[4]:
        safe_render_section("Risk & Drawdown", render_risk_and_drawdown_section, fund_row["isin"], fund_nav, scoring_nav)

    snapshot = build_comparative_snapshot(fund_master)
    selected_snapshot = snapshot.loc[snapshot["fund_name"] == fund_name]
    if not selected_snapshot.empty:
        st.markdown("**Section score summary for this fund**")
        render_selected_fund_score_summary(selected_snapshot.iloc[0].to_dict())


def render_comparative_page(datasets: dict[str, object]) -> None:
    st.subheader("Comparative Insights")
    st.caption("Cross-fund comparison bars, downside RMS analysis, and scenario stress testing versus Nifty 100.")
    fund_master = datasets["fund_master"]

    with st.spinner("Refreshing cross-fund snapshot from Moneycontrol..."):
        snapshot = build_comparative_snapshot(fund_master)

    selected_universe = st.multiselect(
        "Fund universe for comparative analysis",
        options=fund_master["fund_name"].tolist(),
        default=fund_master["fund_name"].tolist(),
        key="ci_universe",
    )
    filtered_snapshot = snapshot.loc[snapshot["fund_name"].isin(selected_universe)].copy()
    filtered_master = fund_master.loc[fund_master["fund_name"].isin(selected_universe)].copy()
    if filtered_snapshot.empty:
        st.info("Select at least one fund in the comparative universe.")
        return

    metric_map = {
        "5Y Return (CAGR)": "return_5y_pct",
        "3Y Return (CAGR)": "return_3y_pct",
        "1Y Return (CAGR)": "return_1y_pct",
        "P/E": "pe",
        "P/B": "pb",
        "Standard Deviation 1Y": "std_dev_1y",
        "Standard Deviation 3Y": "std_dev_3y",
        "Standard Deviation 5Y": "std_dev_5y",
        "Sharpe 1Y": "sharpe_1y",
        "Sharpe 3Y": "sharpe_3y",
        "Sharpe 5Y": "sharpe_5y",
        "Sortino 1Y": "sortino_1y",
        "Sortino 3Y": "sortino_3y",
        "Sortino 5Y": "sortino_5y",
        "Beta 1Y": "beta_1y",
        "Beta 3Y": "beta_3y",
        "Beta 5Y": "beta_5y",
        "Max Drawdown": "max_drawdown_pct",
        "Diversification Score": "diversification_score_raw",
        "Fundamentals Score": "fundamentals_score_raw",
        "Stress Avg Excess Return": "stress_avg_excess_return",
        "Downside RMS": "downside_rms_pct",
    }

    compare_col, stress_col = st.columns([1.05, 1])
    with compare_col:
        metric_label = st.selectbox("Comparison metric", options=list(metric_map.keys()), key="ci_metric")
        render_comparison_bar(filtered_snapshot, metric_map[metric_label], metric_label)
        with st.expander("What this comparison means", expanded=False):
            st.write(
                "- Each bar is one selected large-cap fund from the current universe.\n"
                "- Return metrics are computed from NAV history; risk and fundamentals come from the Moneycontrol API payloads.\n"
                "- Lower-is-better metrics like drawdown or standard deviation are shown as raw values here, not inverted."
            )

        st.markdown("**Downside RMS analyser**")
        rms_start = st.date_input("RMS start date", value=pd.Timestamp("2025-01-01"), key="ci_rms_start")
        rms_end = st.date_input("RMS end date", value=SCORING_CUTOFF_DATE.date(), key="ci_rms_end")
        rms_frame = build_rms_comparison_frame(filtered_master, rms_start, rms_end)
        if not rms_frame.empty:
            render_comparison_bar(rms_frame.rename(columns={"rms_pct": "Downside RMS (%)"}), "Downside RMS (%)", "Downside RMS (%)")
            rms_fund = st.selectbox("Fund for RMS day-mix view", options=filtered_master["fund_name"].tolist(), key="ci_rms_fund")
            rms_selected = rms_frame.loc[rms_frame["fund_name"] == rms_fund]
            if not rms_selected.empty:
                rms_row = rms_selected.iloc[0]
                pie = go.Figure(
                    go.Pie(
                        labels=["Negative days", "Positive days", "Flat days"],
                        values=[
                            rms_row["negative_day_pct"] or 0,
                            rms_row["positive_day_pct"] or 0,
                            rms_row["flat_day_pct"] or 0,
                        ],
                        marker=dict(colors=["#d62728", "#2ca02c", "#bdbdbd"]),
                    )
                )
                pie.update_layout(height=320, title=f"Day Mix for {rms_fund}")
                st.plotly_chart(pie, use_container_width=True)
            with st.expander("How downside RMS is calculated", expanded=False):
                st.write(
                    "- Daily returns are computed from NAV.\n"
                    "- Positive-return days are anchored to `0`.\n"
                    "- Negative-return days are squared, averaged, and square-rooted.\n"
                    "- So a higher downside RMS means a worse pattern of downside shocks."
                )
            st.dataframe(
                rms_frame[["fund_name", "rms_pct", "negative_day_pct", "positive_day_pct", "flat_day_pct"]].sort_values("rms_pct"),
                use_container_width=True,
                hide_index=True,
            )

    with stress_col:
        st.markdown("**Stress Tester**")
        stress_fund = st.selectbox("Fund for stress test", options=filtered_master["fund_name"].tolist(), key="ci_stress_fund")
        scenario_name = st.selectbox("Built-in scenario", options=list(BUILT_IN_SCENARIOS.keys()), key="ci_scenario")
        custom_mode = st.checkbox("Add custom scenario", value=False, key="ci_custom_toggle")
        if custom_mode:
            custom_start = st.date_input("Custom start date", value=pd.Timestamp("2024-09-01"), key="ci_custom_start")
            custom_end = st.date_input("Custom end date", value=pd.Timestamp("2025-03-31"), key="ci_custom_end")
            scenario_start, scenario_end = pd.Timestamp(custom_start), pd.Timestamp(custom_end)
            scenario_label = "Custom Scenario"
        else:
            scenario_start, scenario_end = [pd.Timestamp(value) for value in BUILT_IN_SCENARIOS[scenario_name]]
            scenario_label = scenario_name

        stress_fund_row = fund_master.loc[fund_master["fund_name"] == stress_fund].iloc[0]
        fund_nav = fetch_nav_bundle(stress_fund_row["isin"])["fund_nav"]
        benchmark_nav = fetch_nifty100_series()
        scenario_metrics = compute_scenario_metrics(fund_nav, benchmark_nav, scenario_start, scenario_end)
        render_stress_scenario(stress_fund, scenario_label, scenario_metrics)

    st.markdown("**Cross-fund stress score ranking**")
    stress_ranking = filtered_snapshot[["fund_name", "stress_avg_excess_return", "stress_avg_correlation", "stress_test_score"]].sort_values(
        "stress_test_score",
        ascending=False,
    )
    with st.expander("How to read the stress ranking table", expanded=False):
        st.write(
            "- `stress_avg_excess_return`: average fund return minus Nifty 100 return across the built-in stress scenarios.\n"
            "- `stress_avg_correlation`: average daily-return correlation with Nifty 100 during those scenarios.\n"
            "- `stress_test_score`: normalized score that rewards higher excess return and lower correlation."
        )
    st.dataframe(stress_ranking, use_container_width=True, hide_index=True)


def render_final_ranker_page(datasets: dict[str, object]) -> None:
    st.subheader("Final Ranker")
    st.caption("Allocate explicit weights from `0.0` to `100.0` across metrics, and the app dynamically computes the normalized final score.")
    st.warning(
        "Consistency note: NAV-based score components are capped at 31 May 2026. Holdings, fundamentals, and Moneycontrol risk snapshots are current proxies because historical as-of-May-2026 snapshots are not yet wired."
    )

    fund_master = datasets["fund_master"]
    snapshot = build_comparative_snapshot(fund_master)
    overweight_scores = build_overweight_score_snapshot(datasets)

    score_frames = [
        overweight_scores,
        build_returns_scores(snapshot),
        build_rolling_scores(snapshot),
        build_diversification_scores(snapshot),
        build_fundamentals_scores(snapshot),
        build_risk_scores(snapshot),
        build_drawdown_scores(snapshot),
        build_stress_scores(snapshot),
        build_rms_scores(snapshot),
    ]
    merged_scores = merge_score_frames(score_frames)

    default_metrics = [
        "overweight_1m_score",
        "overweight_3m_score",
        "overweight_6m_score",
        "returns_5y_score",
        "returns_3y_score",
        "returns_1y_score",
        "rollingreturns_5y_score",
        "rollingreturns_3y_score",
        "rollingreturns_1y_score",
        "diversification_score",
        "fundamentals_score",
        "risk_metrics_score",
        "drawdown_score",
        "stress_test_score",
        "rms_downside_score",
    ]

    weight_table = pd.DataFrame(
        {
            "metric": default_metrics,
            "weight_pct": [0.0] * len(default_metrics),
        }
    )
    edited = st.data_editor(
        weight_table,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        key="final_ranker_weights",
    )
    edited["weight_pct"] = pd.to_numeric(edited["weight_pct"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=100.0)
    allocated_weight = float(edited["weight_pct"].sum())
    remaining_weight = 100.0 - allocated_weight
    info_col1, info_col2 = st.columns(2)
    info_col1.metric("Allocated weight", f"{allocated_weight:.2f}%")
    info_col2.metric("Remaining to allocate", f"{remaining_weight:.2f}%")
    if allocated_weight > 100.0:
        st.error("Total allocated weight cannot exceed 100%. Please reduce the inputs.")
        return

    weight_map = dict(zip(edited["metric"], edited["weight_pct"]))

    final_ranking = build_final_ranker(merged_scores, weight_map)
    methodology = build_metric_methodology()

    with st.expander("How final scoring works", expanded=False):
        st.write(
            "- Every raw score is normalized to a 0-100 scale.\n"
            "- You directly allocate weight percentages to the metrics you care about.\n"
            "- If a fund is missing a metric, that metric is excluded from that fund's weight denominator instead of forcing a zero."
        )
        for metric, weight in weight_map.items():
            if weight <= 0:
                continue
            if metric in methodology:
                st.write(f"- `{metric}` ({weight:.2f}%): {methodology[metric]}")

    st.dataframe(final_ranking, use_container_width=True, hide_index=True)
    render_final_ranker_explanation(final_ranking, merged_scores, methodology, weight_map)


def render_nav_chart(fund_nav: pd.DataFrame, fund_name: str) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fund_nav["date"], y=fund_nav["value"], mode="lines", name=fund_name, line=dict(color="#1f77b4")))
    fig.update_layout(
        title="NAV Trend",
        height=440,
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(count=3, label="3Y", step="year", stepmode="backward"),
                    dict(count=5, label="5Y", step="year", stepmode="backward"),
                    dict(step="all", label="Max"),
                ]
            ),
            rangeslider=dict(visible=True),
        ),
        yaxis_title="NAV",
        legend_title="Series",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_return_heatmaps_and_bars(fund_nav: pd.DataFrame) -> None:
    monthly_returns = build_monthly_returns(fund_nav)
    heatmap = build_monthly_heatmap(monthly_returns)
    yearly_returns = build_yearly_returns(fund_nav)
    annual_table = yearly_returns.rename(columns={"year": "period", "return_pct": "absolute_pct"}).copy()
    annual_table["period"] = annual_table["period"].astype(str)

    st.markdown("**Monthly returns heatmap**")
    if not heatmap.empty:
        fig = px.imshow(
            heatmap,
            text_auto=".1f",
            aspect="auto",
            color_continuous_scale=["#d9534f", "#f7f7f7", "#5cb85c"],
        )
        fig.update_layout(height=520, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown("**Monthly chart (simple returns)**")
        if not monthly_returns.empty:
            fig = go.Figure(
                go.Bar(
                    x=monthly_returns["date"].dt.strftime("%b-%Y"),
                    y=monthly_returns["monthly_return_pct"],
                    marker_color=["#2ca02c" if value >= 0 else "#d62728" for value in monthly_returns["monthly_return_pct"]],
                )
            )
            fig.update_layout(height=380, xaxis_tickangle=-90, yaxis_title="Monthly return (%)")
            st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("**Yearly chart**")
        if not yearly_returns.empty:
            fig = go.Figure(
                go.Bar(
                    x=yearly_returns["year"].astype(str),
                    y=yearly_returns["return_pct"],
                    text=yearly_returns["return_pct"].map(lambda value: f"{value:.1f}%"),
                    textposition="outside",
                    marker_color=["#2ca02c" if value >= 0 else "#d62728" for value in yearly_returns["return_pct"]],
                )
            )
            fig.update_layout(height=380, yaxis_title="Calendar-year return (%)")
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("Calendar-year returns table", expanded=False):
        st.write(
            "`absolute_pct` is the fund's compounded calendar-year return from monthly NAV changes. "
            "The category proxy comparison is removed here to keep this view clean and avoid misleading comparisons."
        )
        st.dataframe(annual_table, use_container_width=True, hide_index=True)


def render_rolling_returns_section(isin: str) -> None:
    base_payload = fetch_rolling_returns(isin)
    rolling_options = base_payload["available_periods"] or [base_payload["duration"]]
    rolling_index = rolling_options.index(base_payload["duration"]) if base_payload["duration"] in rolling_options else 0
    duration = st.selectbox(
        "Rolling return duration",
        options=rolling_options,
        index=rolling_index,
        key="fa_rolling_duration",
    )
    payload = fetch_rolling_returns(isin, duration)
    rolling_returns = payload["returns"]
    summary = build_rolling_summary(rolling_returns)

    cols = st.columns(4)
    cols[0].metric("Mean CAGR", format_pct(summary["mean_cagr"]))
    cols[1].metric("Median CAGR", format_pct(summary["median_cagr"]))
    cols[2].metric("Best CAGR", format_pct(summary["best_cagr"]))
    cols[3].metric("Worst CAGR", format_pct(summary["worst_cagr"]))

    if not rolling_returns.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rolling_returns["end_date"], y=rolling_returns["cagr_val"], mode="lines", name="Rolling CAGR"))
        fig.update_layout(height=400, yaxis_title="Rolling CAGR (%)", title=f"Rolling Returns - {duration}")
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Rolling returns table", expanded=False):
            st.dataframe(rolling_returns, use_container_width=True, hide_index=True)


def render_portfolio_section(isin: str) -> None:
    payload = fetch_portfolio(isin)
    summary = payload["summary"]
    concentration = summary.get("concentration", {})
    holdings = payload["holdings"]

    metric_cols = st.columns(4)
    metric_cols[0].metric("Avg. Market Cap", str(concentration.get("avg_market_cap", "NA")).replace("â¹", "₹"))
    metric_cols[1].metric("Top 10 Weight", format_pct(safe_float(concentration.get("top_10_stk_wt"))))
    metric_cols[2].metric("Top 5 Weight", format_pct(safe_float(concentration.get("top_5_stk_wt"))))
    metric_cols[3].metric("Number of Holdings", str(concentration.get("number_of_holding", "NA")))

    st.markdown("**Top 5 equity holdings**")
    if not holdings.empty:
        top_holdings = holdings.sort_values("weighting", ascending=False).head(5).copy()
        st.dataframe(top_holdings, use_container_width=True, hide_index=True)

    with st.expander("How diversification is interpreted", expanded=False):
        st.write(
            "- Higher number of holdings generally means broader diversification.\n"
            "- Lower `top_10_stk_wt` and `top_5_stk_wt` usually mean the fund is less concentrated.\n"
            "- These snapshot fields are used later for the diversification score."
        )


def render_fundamentals_section(isin: str) -> None:
    fundamentals = fetch_fundamentals(isin)
    table = build_fundamentals_table(fundamentals)
    with st.expander("How to read fundamentals", expanded=False):
        st.write(
            "- Valuation ratios such as P/E, P/B, Price/Sales, and Price/Cash Flow are generally better when lower relative to peers.\n"
            "- Dividend Yield and ROE are generally better when higher, though not in isolation."
        )
    st.dataframe(table, use_container_width=True, hide_index=True)


def render_risk_and_drawdown_section(isin: str, fund_nav: pd.DataFrame, scoring_nav: pd.DataFrame) -> None:
    risk_payload = fetch_risk_metrics(isin)
    horizon = st.selectbox("Risk horizon", options=["1y", "3y", "5y"], format_func=str.upper, key="fa_risk_horizon")
    risk_table = build_risk_table(risk_payload, horizon)
    st.dataframe(risk_table, use_container_width=True, hide_index=True)

    st.markdown("**Top drawdowns**")
    drawdown_series = build_drawdown_series(scoring_nav if not scoring_nav.empty else fund_nav)
    drawdown_table = calculate_drawdown_table(scoring_nav if not scoring_nav.empty else fund_nav)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fund_nav["date"], y=fund_nav["value"], mode="lines", name="NAV", line=dict(color="#d62728")))
    for row in drawdown_table.itertuples(index=False):
        fig.add_vrect(x0=row.start_date, x1=row.end_date, fillcolor="#b0b0b0", opacity=0.25, line_width=0)
    fig.update_layout(height=420, title="NAV with Top Drawdown Windows Marked", yaxis_title="NAV")
    st.plotly_chart(fig, use_container_width=True)

    if not drawdown_table.empty:
        display = drawdown_table.copy()
        display["loss_pct"] = display["loss_pct"].map(lambda value: f"{value:.2f}%")
        display["recovery_date"] = display["recovery_date"].dt.strftime("%Y-%m-%d").fillna("Still ongoing")
        display["days_to_recover"] = display["days_to_recover"].fillna("Still ongoing")
        display["recovered_fraction"] = display["recovered_fraction"].map(lambda value: f"{value:.0%}")
        st.dataframe(display, use_container_width=True, hide_index=True)

    with st.expander("How drawdowns are computed", expanded=False):
        st.write(
            f"- Drawdown ranking for scoring uses NAV data only up to `{SCORING_CUTOFF_DATE.date()}`.\n"
            "- `loss_pct` is peak-to-trough decline.\n"
            "- `days_to_recover` is valley-to-recovery time; if not recovered, the row remains open."
        )


def render_selected_fund_score_summary(row: dict[str, Any]) -> None:
    metrics = {
        "Returns Proxy": row.get("returns_score_blend"),
        "Rolling Proxy": row.get("rolling_score_blend"),
        "Diversification": row.get("diversification_score"),
        "Fundamentals": row.get("fundamentals_score"),
        "Risk": row.get("risk_score_blend"),
        "Drawdown": row.get("drawdown_score"),
        "Stress": row.get("stress_test_score"),
        "RMS": row.get("rms_downside_score"),
    }
    columns = st.columns(len(metrics))
    for column, (label, value) in zip(columns, metrics.items()):
        column.metric(label, format_pct(value, suffix=" / 100"))


def render_comparison_bar(snapshot: pd.DataFrame, metric_column: str, metric_label: str) -> None:
    frame = snapshot[["fund_name", metric_column]].dropna().sort_values(metric_column, ascending=False).copy()
    is_rms_metric = "RMS" in metric_label.upper()
    text_values = (
        frame[metric_column].map(lambda value: f"{value:.2f}%")
        if is_rms_metric
        else frame[metric_column].map(lambda value: f"{value:.2f}")
    )
    fig = go.Figure(
        go.Bar(
            x=frame["fund_name"],
            y=frame[metric_column],
            text=text_values,
            textposition="outside",
            marker_color="#4e79a7",
        )
    )
    fig.update_layout(height=480, title=metric_label, xaxis_tickangle=-70, yaxis_title=metric_label)
    st.plotly_chart(fig, use_container_width=True)


def render_stress_scenario(fund_name: str, scenario_label: str, scenario_metrics: dict[str, Any]) -> None:
    plot_frame = scenario_metrics["plot_frame"].copy()
    st.markdown(f"**{scenario_label}**")
    if not plot_frame.empty and {"value_fund", "value_benchmark"}.issubset(plot_frame.columns):
        if plot_frame["value_fund"].notna().any() and plot_frame["value_benchmark"].notna().any():
            plot_frame["fund_indexed"] = plot_frame["value_fund"] / plot_frame["value_fund"].iloc[0] * 100
            plot_frame["benchmark_indexed"] = plot_frame["value_benchmark"] / plot_frame["value_benchmark"].iloc[0] * 100
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=plot_frame["date"], y=plot_frame["fund_indexed"], mode="lines", name=fund_name))
            fig.add_trace(go.Scatter(x=plot_frame["date"], y=plot_frame["benchmark_indexed"], mode="lines", name="Nifty 100"))
            fig.update_layout(height=360, yaxis_title="Indexed to 100", title="Scenario Path Comparison")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Benchmark data was unavailable for this window, so only the fund series can be shown.")
    else:
        st.info("No overlapping benchmark data was available for this scenario window.")

    summary = pd.DataFrame(
        [
            {"metric": "Start Date", "value": scenario_metrics["start_date"].date()},
            {"metric": "End Date", "value": scenario_metrics["end_date"].date()},
            {"metric": "Excess Return", "value": format_pct(scenario_metrics["excess_return_pct"])},
            {"metric": "Correlation", "value": format_number(scenario_metrics["correlation"])},
            {"metric": "Fund Return", "value": format_pct(scenario_metrics["fund_return_pct"])},
            {"metric": "Benchmark Return", "value": format_pct(scenario_metrics["benchmark_return_pct"])},
            {"metric": "Fund Volatility", "value": format_pct(scenario_metrics["fund_volatility_pct"])},
            {"metric": "Benchmark Volatility", "value": format_pct(scenario_metrics["benchmark_volatility_pct"])},
        ]
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)


@st.cache_data(show_spinner=False, ttl=21600)
def build_comparative_snapshot(fund_master: pd.DataFrame) -> pd.DataFrame:
    benchmark_nav = filter_to_cutoff(fetch_nifty100_series())
    rows: list[dict[str, Any]] = []

    for fund in fund_master.itertuples(index=False):
        try:
            nav_bundle = fetch_nav_bundle(fund.isin)
            fund_nav_full = nav_bundle["fund_nav"]
            fund_nav_cutoff = filter_to_cutoff(fund_nav_full)
            portfolio = fetch_portfolio(fund.isin)
            fundamentals = fetch_fundamentals(fund.isin)
            risk = fetch_risk_metrics(fund.isin)

            rolling_1y = fetch_rolling_returns(fund.isin, "1Y")["returns"]
            rolling_3y = fetch_rolling_returns(fund.isin, "3Y")["returns"]
            rolling_5y = fetch_rolling_returns(fund.isin, "5Y")["returns"]
            rolling_1y_summary = build_rolling_summary(rolling_1y)
            rolling_3y_summary = build_rolling_summary(rolling_3y)
            rolling_5y_summary = build_rolling_summary(rolling_5y)

            drawdown_table = calculate_drawdown_table(fund_nav_cutoff)
            drawdown_loss = float(drawdown_table["loss_pct"].min()) if not drawdown_table.empty else None
            recovery_days = float(drawdown_table["days_to_recover"].dropna().mean()) if not drawdown_table.empty else None
            rms_metrics = compute_downside_rms(
                fund_nav_cutoff,
                fund_nav_cutoff["date"].min() if not fund_nav_cutoff.empty else SCORING_CUTOFF_DATE,
                SCORING_CUTOFF_DATE,
            )

            diversification = calculate_diversification_snapshot(portfolio)

            scenario_rows = []
            for scenario_name, (start_date, end_date) in BUILT_IN_SCENARIOS.items():
                metrics = compute_scenario_metrics(fund_nav_cutoff, benchmark_nav, start_date, end_date)
                scenario_rows.append(
                    {
                        "scenario": scenario_name,
                        "excess_return_pct": metrics["excess_return_pct"],
                        "correlation": metrics["correlation"],
                    }
                )
            scenario_frame = pd.DataFrame(scenario_rows)

            row = {
                "fund_name": fund.fund_name,
                "isin": fund.isin,
                "coarse_fund_key": coarse_fund_key(fund.fund_name),
                "return_1y_pct": calculate_cagr(fund_nav_cutoff, 1, SCORING_CUTOFF_DATE),
                "return_3y_pct": calculate_cagr(fund_nav_cutoff, 3, SCORING_CUTOFF_DATE),
                "return_5y_pct": calculate_cagr(fund_nav_cutoff, 5, SCORING_CUTOFF_DATE),
                "return_10y_pct": calculate_cagr(fund_nav_cutoff, 10, SCORING_CUTOFF_DATE),
                "rolling_1y_median_cagr": rolling_1y_summary["median_cagr"],
                "rolling_3y_median_cagr": rolling_3y_summary["median_cagr"],
                "rolling_5y_median_cagr": rolling_5y_summary["median_cagr"],
                "pe": safe_float(fundamentals.get("pe")),
                "pb": safe_float(fundamentals.get("pb")),
                "price_sale": safe_float(fundamentals.get("price_sale")),
                "price_cash_flow": safe_float(fundamentals.get("price_cash_flow")),
                "dividend_yield": safe_float(fundamentals.get("dividend_yield")),
                "roe": safe_float(fundamentals.get("roe")),
                "std_dev_1y": safe_float(risk.get("risk_std_dev", {}).get("1y")),
                "std_dev_3y": safe_float(risk.get("risk_std_dev", {}).get("3y")),
                "std_dev_5y": safe_float(risk.get("risk_std_dev", {}).get("5y")),
                "sharpe_1y": safe_float(risk.get("sharpe_ratio", {}).get("1y")),
                "sharpe_3y": safe_float(risk.get("sharpe_ratio", {}).get("3y")),
                "sharpe_5y": safe_float(risk.get("sharpe_ratio", {}).get("5y")),
                "sortino_1y": safe_float(risk.get("sortino_ratio", {}).get("1y")),
                "sortino_3y": safe_float(risk.get("sortino_ratio", {}).get("3y")),
                "sortino_5y": safe_float(risk.get("sortino_ratio", {}).get("5y")),
                "beta_1y": safe_float(risk.get("beta", {}).get("1y")),
                "beta_3y": safe_float(risk.get("beta", {}).get("3y")),
                "beta_5y": safe_float(risk.get("beta", {}).get("5y")),
                "max_drawdown_pct": drawdown_loss,
                "avg_drawdown_recovery_days": recovery_days,
                "number_of_holdings": diversification.get("number_of_holdings"),
                "top_10_weight_pct": diversification.get("top_10_weight_pct"),
                "top_5_weight_pct": diversification.get("top_5_weight_pct"),
                "top_3_sector_weight_pct": diversification.get("top_3_sector_weight_pct"),
                "diversification_score_raw": np.nan,
                "fundamentals_score_raw": np.nan,
                "stress_avg_excess_return": scenario_frame["excess_return_pct"].mean() if not scenario_frame.empty else None,
                "stress_avg_correlation": scenario_frame["correlation"].mean() if not scenario_frame.empty else None,
                "downside_rms_pct": rms_metrics["rms_pct"],
            }
            rows.append(row)
        except Exception:
            rows.append({"fund_name": fund.fund_name, "isin": fund.isin, "coarse_fund_key": coarse_fund_key(fund.fund_name)})

    snapshot = pd.DataFrame(rows)
    if snapshot.empty:
        return snapshot

    expected_columns = [
        "return_1y_pct",
        "return_3y_pct",
        "return_5y_pct",
        "return_10y_pct",
        "rolling_1y_median_cagr",
        "rolling_3y_median_cagr",
        "rolling_5y_median_cagr",
        "pe",
        "pb",
        "price_sale",
        "price_cash_flow",
        "dividend_yield",
        "roe",
        "std_dev_1y",
        "std_dev_3y",
        "std_dev_5y",
        "sharpe_1y",
        "sharpe_3y",
        "sharpe_5y",
        "sortino_1y",
        "sortino_3y",
        "sortino_5y",
        "beta_1y",
        "beta_3y",
        "beta_5y",
        "max_drawdown_pct",
        "avg_drawdown_recovery_days",
        "number_of_holdings",
        "top_10_weight_pct",
        "top_5_weight_pct",
        "top_3_sector_weight_pct",
        "stress_avg_excess_return",
        "stress_avg_correlation",
        "downside_rms_pct",
    ]
    for column in expected_columns:
        if column not in snapshot.columns:
            snapshot[column] = np.nan

    diversification_scores = build_diversification_scores(snapshot)
    fundamentals_scores = build_fundamentals_scores(snapshot)
    risk_scores = build_risk_scores(snapshot)
    drawdown_scores = build_drawdown_scores(snapshot)
    stress_scores = build_stress_scores(snapshot)
    rms_scores = build_rms_scores(snapshot)
    returns_scores = build_returns_scores(snapshot)
    rolling_scores = build_rolling_scores(snapshot)

    join_keys = ["fund_name", "coarse_fund_key"]
    snapshot = snapshot.merge(diversification_scores, on=join_keys, how="left", validate="one_to_one")
    snapshot = snapshot.merge(fundamentals_scores, on=join_keys, how="left", validate="one_to_one")
    snapshot = snapshot.merge(risk_scores, on=join_keys, how="left", validate="one_to_one")
    snapshot = snapshot.merge(drawdown_scores, on=join_keys, how="left", validate="one_to_one")
    snapshot = snapshot.merge(stress_scores, on=join_keys, how="left", validate="one_to_one")
    snapshot = snapshot.merge(rms_scores, on=join_keys, how="left", validate="one_to_one")
    snapshot = snapshot.merge(returns_scores, on=join_keys, how="left", validate="one_to_one")
    snapshot = snapshot.merge(rolling_scores, on=join_keys, how="left", validate="one_to_one")

    snapshot["diversification_score_raw"] = snapshot["diversification_score"]
    snapshot["fundamentals_score_raw"] = snapshot["fundamentals_score"]
    snapshot["returns_score_blend"] = snapshot[["returns_1y_score", "returns_3y_score", "returns_5y_score"]].mean(axis=1, skipna=True)
    snapshot["rolling_score_blend"] = snapshot[["rollingreturns_1y_score", "rollingreturns_3y_score", "rollingreturns_5y_score"]].mean(axis=1, skipna=True)
    snapshot["risk_score_blend"] = snapshot["risk_metrics_score"]
    snapshot["stress_score"] = snapshot["stress_test_score"]
    return snapshot


@st.cache_data(show_spinner=False)
def build_overweight_score_snapshot(datasets: dict[str, object]) -> pd.DataFrame:
    frames = []
    for lookback_label, target_column in [("1M", "overweight_1m_score"), ("3M", "overweight_3m_score"), ("6M", "overweight_6m_score")]:
        try:
            result = compute_overweight_score(
                fund_weights=datasets["fund_weights"],
                sector_returns=datasets["sector_returns"],
                sector_mapping=datasets["sector_mapping"],
                price_history=datasets["price_history"],
                lookback_label=lookback_label,
            )
            frame = result["rankings"][["fund_name", "total_score"]].copy()
            frame["coarse_fund_key"] = frame["fund_name"].map(coarse_fund_key)
            frame[target_column] = normalize_series(frame["total_score"], higher_is_better=True)
            frames.append(frame[["fund_name", "coarse_fund_key", target_column]])
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=["fund_name", "coarse_fund_key"])
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["fund_name", "coarse_fund_key"], how="outer")
    return merged


def safe_render_section(title: str, render_func, *args) -> None:
    try:
        render_func(*args)
    except Exception as exc:
        st.error(f"{title} could not be loaded for this fund: {exc}")
        st.info("This section is isolated, so the rest of the analytics page continues to work.")


@st.cache_data(show_spinner=False, ttl=21600)
def build_rms_comparison_frame(
    fund_master: pd.DataFrame,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fund in fund_master.itertuples(index=False):
        try:
            nav = fetch_nav_bundle(fund.isin)["fund_nav"]
            rms = compute_downside_rms(nav, start_date, end_date)
            rows.append(
                {
                    "fund_name": fund.fund_name,
                    "rms_pct": rms["rms_pct"],
                    "negative_day_pct": rms["negative_day_pct"],
                    "positive_day_pct": rms["positive_day_pct"],
                    "flat_day_pct": rms["flat_day_pct"],
                }
            )
        except Exception:
            continue
    return pd.DataFrame(rows)


def render_final_ranker_explanation(
    final_ranking: pd.DataFrame,
    merged_scores: pd.DataFrame,
    methodology: dict[str, str],
    weight_map: dict[str, float],
) -> None:
    if final_ranking.empty:
        return
    winner = final_ranking.iloc[0]
    winner_scores = merged_scores.loc[merged_scores["fund_name"] == winner["fund_name"]]
    if winner_scores.empty:
        return
    score_row = winner_scores.iloc[0]
    active_weights = {metric: weight for metric, weight in weight_map.items() if weight > 0 and metric in score_row.index}
    if not active_weights:
        return

    st.markdown("**Why the top-ranked fund scored this way**")
    st.write(f"`{winner['fund_name']}` is rank 1 with a final normalized score of `{winner['final_score']:.2f}`.")
    explanation_rows = []
    total_weight = sum(active_weights.values())
    for metric, raw_weight in active_weights.items():
        metric_value = score_row.get(metric)
        normalized_weight = raw_weight / total_weight if total_weight > 0 else 0.0
        contribution = None if pd.isna(metric_value) else metric_value * normalized_weight
        explanation_rows.append(
            {
                "metric": metric,
                "assigned_weight_pct": raw_weight,
                "effective_weight_pct": normalized_weight * 100,
                "raw_score": metric_value,
                "weighted_contribution": contribution,
                "method": methodology.get(metric, ""),
            }
        )
    st.dataframe(pd.DataFrame(explanation_rows), use_container_width=True, hide_index=True)


def render_lookback_section(lookback_label: str, module_result: dict[str, object]) -> None:
    summary = module_result["summary"]
    focus_sectors = module_result["focus_sectors"]
    rankings = module_result["rankings"]
    sector_details = module_result["sector_details"]

    st.divider()
    st.subheader(f"{lookback_label} Sector Rotation")
    st.caption(
        f"Returns used: {summary['return_start'].date()} to {summary['return_end'].date()} | "
        f"Pre-positioning window used: {summary['signal_start_date'].date()} to {summary['signal_end_date'].date()}"
    )
    st.write(
        f"For `{lookback_label}`, the return ranking comes from `stock_performance_ranks1.csv` "
        f"between `{summary['return_start'].date()}` and `{summary['return_end'].date()}`. "
        f"To test whether funds moved early, the scorer compares weights between "
        f"`{summary['signal_start_date'].date()}` and `{summary['signal_end_date'].date()}`, "
        f"the two monthly disclosures immediately before the return window begins."
    )
    with st.expander("Why these weight-change dates are used", expanded=False):
        st.write(
            "- The engine uses the **latest available weight change fully before the return window begins**.\n"
            f"- So for `{lookback_label}`, if returns start on `{summary['return_start'].date()}`, the chosen signal pair is `{summary['signal_start_date'].date()}` to `{summary['signal_end_date'].date()}`.\n"
            "- This is intentional: it tests whether the manager was already positioned before the move started.\n"
            "- To reduce the blind spot where a sector stays flat for much of the window and only moves late, the scorer now also uses a rolling pre-positioning component based on raw daily index prices from `PRICES.xlsx`."
        )

    col1, col2, col3 = st.columns(3)
    col1.metric("Funds scored", int(summary["fund_count"]))
    col2.metric("Sectors scored", int(summary["sector_count"]))
    col3.metric("Weight change dates", f"{summary['signal_start_date'].date()} -> {summary['signal_end_date'].date()}")

    if module_result["issues"]:
        st.warning("Intelligent replacements / unresolved items: " + "; ".join(module_result["issues"]))

    render_sector_chart(focus_sectors, lookback_label)
    render_sector_tables(focus_sectors)
    render_rankings(rankings, sector_details, lookback_label)
    render_detail_table(sector_details)
    render_notes(module_result["notes"])


def render_sector_chart(focus_sectors: pd.DataFrame, lookback_label: str) -> None:
    top = focus_sectors.loc[focus_sectors["bucket"] == "Top 5"].copy()
    bottom = focus_sectors.loc[focus_sectors["bucket"] == "Bottom 5"].copy()

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Top 5 Winners", "Top 5 Losers"),
        horizontal_spacing=0.14,
    )
    fig.add_trace(
        go.Bar(
            x=top["Stock"],
            y=top["Return"],
            marker_color="#94D3A2",
            text=top["Return"].map(lambda value: f"{value:.2f}%"),
            textposition="outside",
            name="Top 5",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=bottom["Stock"],
            y=bottom["Return"],
            marker_color="#F19A9A",
            text=bottom["Return"].map(lambda value: f"{value:.2f}%"),
            textposition="outside",
            name="Bottom 5",
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        title=f"Top and Bottom Sector/Index Performance for {lookback_label}",
        height=460,
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=70, b=40, l=20, r=20),
    )
    fig.update_yaxes(title_text="Aggregated return (%)", zeroline=True, zerolinecolor="#666")
    st.plotly_chart(fig, use_container_width=True)


def render_sector_tables(focus_sectors: pd.DataFrame) -> None:
    left, right = st.columns(2)
    with left:
        st.markdown("**Top 5 winners**")
        st.dataframe(
            focus_sectors.loc[focus_sectors["bucket"] == "Top 5", ["Stock", "Return", "Rank"]],
            use_container_width=True,
            hide_index=True,
        )
    with right:
        st.markdown("**Top 5 losers**")
        st.dataframe(
            focus_sectors.loc[focus_sectors["bucket"] == "Bottom 5", ["Stock", "Return", "Rank"]],
            use_container_width=True,
            hide_index=True,
        )


def render_rankings(rankings: pd.DataFrame, sector_details: pd.DataFrame, lookback_label: str) -> None:
    display = rankings.copy()
    display.insert(0, "rank", range(1, len(display) + 1))
    st.markdown("**Fund ranking for this timeline**")
    st.caption(
        "Higher `total_score` means the fund made more favorable pre-positioning moves across the 10 focus sectors for this lookback."
    )
    with st.expander("How to read the ranking columns", expanded=False):
        st.write(
            "- `total_score`: sum of all sector scores for this fund across the 10 focus sectors.\n"
            "- `favorable_hits`: number of sectors where the fund moved the right way before the move.\n"
            "- `unfavorable_hits`: number of sectors where it moved the wrong way.\n"
            "- `avg_aligned_delta_pct`: average weight-change signal after direction adjustment.\n"
            "- `best_sector` / `worst_sector`: the strongest positive and negative sector contributors."
        )
        st.write(
            "- `sector_score` now blends two components:\n"
            "  1. latest pre-window positioning score\n"
            "  2. rolling pre-positioning score across the monthly subwindows inside the selected return horizon.\n"
            "- Current blend: `50% latest signal + 50% rolling signal`.\n"
            "- `aligned_delta_pct` is positive when the fund moved in the favorable direction before the sector move."
        )
    st.dataframe(
        display[
            [
                "rank",
                "fund_name",
                "total_score",
                "favorable_hits",
                "unfavorable_hits",
                "avg_aligned_delta_pct",
                "best_sector",
                "worst_sector",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    with st.expander(f"Open fund notes for {lookback_label}", expanded=False):
        st.write(
            "Use this to inspect one fund at a time across the 10 focus sectors for this timeline. "
            "Green rows mean the fund moved in the favorable direction before the sector move; red rows mean it did not."
        )
        selected_universe = st.multiselect(
            f"Fund universe for {lookback_label}",
            options=display["fund_name"].tolist(),
            default=display["fund_name"].tolist(),
            key=f"ow_universe_{lookback_label}",
        )
        fund_options = [fund_name for fund_name in display["fund_name"].tolist() if fund_name in selected_universe]
        if not fund_options:
            st.info("Select at least one fund in this universe.")
            return
        selected_fund = st.selectbox(
            f"Choose a fund for {lookback_label}",
            options=fund_options,
            key=f"fund_note_{lookback_label}",
        )

        if selected_fund:
            fund_rows = (
                sector_details.loc[sector_details["fund_name"] == selected_fund]
                .sort_values(["bucket", "sector_score"], ascending=[True, False])
                .copy()
            )
            score_row = display.loc[display["fund_name"] == selected_fund].iloc[0]
            st.markdown(
                f"**{selected_fund}** - total_score `{score_row['total_score']:.2f}`, "
                f"favorable hits `{int(score_row['favorable_hits'])}`, "
                f"unfavorable hits `{int(score_row['unfavorable_hits'])}`"
            )
            note_view = fund_rows[
                [
                    "index_name",
                    "bucket",
                    "sector_return_pct",
                    "mapped_weight_sectors",
                    "signal_start_date",
                    "signal_end_date",
                    "start_weight_pct",
                    "end_weight_pct",
                    "delta_weight_pct",
                    "aligned_delta_pct",
                    "sector_score",
                    "favorable_move",
                ]
            ].copy()
            note_view = note_view.rename(
                columns={
                    "index_name": "sector/index",
                    "bucket": "winner_or_loser_group",
                    "signal_start_date": "weight_from",
                    "signal_end_date": "weight_to",
                    "start_weight_pct": "weight_from_pct",
                    "end_weight_pct": "weight_to_pct",
                    "delta_weight_pct": "delta_pct",
                    "latest_aligned_delta_pct": "latest_aligned_delta_pct",
                    "rolling_aligned_delta_pct": "rolling_aligned_delta_pct",
                    "favorable_move": "favorable",
                }
            )
            styled = note_view.style.apply(
                lambda row: [
                    (
                        "background-color: #D8F3DC"
                        if bool(row["favorable"])
                        else "background-color: #F8D7DA"
                    )
                    if col in {"delta_pct", "aligned_delta_pct", "latest_aligned_delta_pct", "rolling_aligned_delta_pct", "sector_score"}
                    else ""
                    for col in row.index
                ],
                axis=1,
            ).format(
                {
                    "sector_return_pct": "{:.2f}",
                    "weight_from_pct": "{:.2f}",
                    "weight_to_pct": "{:.2f}",
                    "delta_pct": "{:+.2f}",
                    "aligned_delta_pct": "{:+.2f}",
                    "latest_aligned_delta_pct": "{:+.2f}",
                    "rolling_aligned_delta_pct": "{:+.2f}",
                    "sector_score": "{:+.2f}",
                }
            )
            st.dataframe(styled, use_container_width=True, hide_index=True)


def render_detail_table(sector_details: pd.DataFrame) -> None:
    detail_view = sector_details[
        [
            "fund_name",
            "index_name",
            "bucket",
            "sector_return_pct",
            "mapped_weight_sectors",
            "signal_start_date",
            "signal_end_date",
            "start_weight_pct",
            "end_weight_pct",
            "delta_weight_pct",
            "aligned_delta_pct",
            "latest_aligned_delta_pct",
            "rolling_aligned_delta_pct",
            "rolling_weighted_alignment",
            "sector_score",
            "favorable_move",
        ]
    ].copy()

    def style_row(row: pd.Series) -> list[str]:
        color = "#D8F3DC" if bool(row["favorable_move"]) else "#F8D7DA"
        styles = ["" for _ in row.index]
        for column in ["delta_weight_pct", "aligned_delta_pct", "latest_aligned_delta_pct", "rolling_aligned_delta_pct", "rolling_weighted_alignment", "sector_score"]:
            idx = row.index.get_loc(column)
            styles[idx] = f"background-color: {color}"
        return styles

    styled = detail_view.style.apply(style_row, axis=1).format(
        {
            "sector_return_pct": "{:.2f}",
            "start_weight_pct": "{:.2f}",
            "end_weight_pct": "{:.2f}",
            "delta_weight_pct": "{:+.2f}",
            "aligned_delta_pct": "{:+.2f}",
            "latest_aligned_delta_pct": "{:+.2f}",
            "rolling_aligned_delta_pct": "{:+.2f}",
            "rolling_weighted_alignment": "{:+.2f}",
            "sector_score": "{:+.2f}",
        }
    )

    with st.expander("Fund-by-fund sector detail", expanded=False):
        st.write(
            "Each row is one fund against one focus sector for the selected timeline. "
            "`signal_start_date` and `signal_end_date` are the two disclosure dates used to measure the pre-move weight change. "
            "`aligned_delta_pct` flips the sign when the sector later fell, so positive values always mean the fund moved in the favorable direction."
        )
        st.write(
            "`mapped_weight_sectors` shows which industry labels from the raw fund files were matched to the chosen index/sector. "
            "`start_weight_pct` and `end_weight_pct` are summed mapped exposures on the two signal dates. "
            "`delta_weight_pct` is the raw latest pre-window change. "
            "`rolling_aligned_delta_pct` is the signed sum across monthly subwindows inside the selected horizon. "
            "`rolling_weighted_alignment` additionally weights those monthly moves by the absolute monthly sector return, so late spikes or falls can still reward timely in-window build-up."
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)


def render_notes(notes: list[str]) -> None:
    with st.expander("Method notes", expanded=False):
        for note in notes:
            st.write(f"- {note}")


def render_run_status(issues: dict[str, str]) -> None:
    st.divider()
    st.subheader("Run status")
    if issues:
        st.warning("Some timeline runs had issues.")
        st.json(issues)
    else:
        st.success("All selected timelines completed successfully.")


def parse_lookback_months(label: str) -> int:
    return int(str(label).replace("M", "").strip())


def format_pct(value: float | None, suffix: str = "%") -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{value:.2f}{suffix}"


def format_number(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{value:.2f}"


if __name__ == "__main__":
    main()
