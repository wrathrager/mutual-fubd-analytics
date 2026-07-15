from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd


ModuleFunc = Callable[..., dict[str, Any] | None]


@dataclass(frozen=True)
class ScoreModule:
    name: str
    weight: float
    compute: ModuleFunc


def run_score_modules(
    modules: list[ScoreModule],
    **kwargs: Any,
) -> dict[str, Any]:
    module_outputs: dict[str, dict[str, Any]] = {}
    combined_frames: list[pd.DataFrame] = []
    errors: dict[str, str] = {}

    for module in modules:
        try:
            result = module.compute(**kwargs)
            if not result:
                continue

            rankings = result.get("rankings")
            if isinstance(rankings, pd.DataFrame) and not rankings.empty:
                score_column = result.get("score_column")
                frame = rankings.copy()
                if score_column and score_column in frame.columns:
                    weighted_column = f"{module.name}_weighted_score"
                    frame[weighted_column] = frame[score_column] * module.weight
                combined_frames.append(frame)

            module_outputs[module.name] = result
        except Exception as exc:
            errors[module.name] = str(exc)

    composite = _combine_frames(combined_frames)
    return {
        "module_outputs": module_outputs,
        "composite_rankings": composite,
        "errors": errors,
    }


def _combine_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()

    composite = frames[0].copy()
    for frame in frames[1:]:
        shared_cols = [column for column in ("fund_name",) if column in frame.columns]
        merge_cols = shared_cols + [
            column
            for column in frame.columns
            if column.endswith("_weighted_score")
        ]
        composite = composite.merge(
            frame[merge_cols],
            on="fund_name",
            how="outer",
        )

    weighted_columns = [
        column for column in composite.columns if column.endswith("_weighted_score")
    ]
    if weighted_columns:
        composite["composite_score"] = composite[weighted_columns].sum(axis=1, skipna=True)
        composite = composite.sort_values("composite_score", ascending=False)

    return composite.reset_index(drop=True)
