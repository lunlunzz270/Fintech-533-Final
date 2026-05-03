from __future__ import annotations

import numpy as np
import pandas as pd

import config


def assign_quantiles(scores: pd.Series, num_quantiles: int) -> pd.Series:
    """Assign quantiles safely using ranked scores."""
    valid = scores.dropna()
    if len(valid) < num_quantiles or valid.nunique() < 2:
        return pd.Series(np.nan, index=scores.index)
    ranked = valid.rank(method="first")
    quantiles = pd.qcut(ranked, q=num_quantiles, labels=False) + 1
    result = pd.Series(np.nan, index=scores.index)
    result.loc[valid.index] = quantiles.astype(float)
    return result


def compute_quantile_returns(
    factor_data: pd.DataFrame,
    score_column: str = "combined_score",
    num_quantiles: int = config.NUM_QUANTILES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute daily and summary quantile returns."""
    rows: list[dict[str, float | str]] = []

    for date, daily in factor_data.groupby("date", sort=True):
        eligible = daily.dropna(subset=[score_column, "forward_return"]).copy()
        if len(eligible) < config.MIN_CROSS_SECTION_SIZE:
            continue

        eligible["quantile"] = assign_quantiles(eligible[score_column], num_quantiles)
        eligible = eligible.dropna(subset=["quantile"])
        if eligible.empty:
            continue

        grouped = eligible.groupby("quantile")["forward_return"].mean()
        row: dict[str, float | str] = {"date": date}
        for quantile in range(1, num_quantiles + 1):
            row[f"Q{quantile}"] = float(grouped.get(float(quantile), np.nan))
        if all(f"Q{quantile}" in row for quantile in (1, num_quantiles)):
            row[f"Q{num_quantiles}_minus_Q1"] = row[f"Q{num_quantiles}"] - row["Q1"]
        rows.append(row)

    daily_quantiles = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    summary_rows: list[dict[str, float | str]] = []
    for column in daily_quantiles.columns:
        if column == "date":
            continue
        series = daily_quantiles[column].dropna()
        if series.empty:
            summary_rows.append(
                {
                    "quantile": column,
                    "mean_daily_return": np.nan,
                    "annualized_return": np.nan,
                    "annualized_volatility": np.nan,
                    "sharpe": np.nan,
                    "cumulative_return": np.nan,
                }
            )
            continue
        daily_std = series.std(ddof=1)
        summary_rows.append(
            {
                "quantile": column,
                "mean_daily_return": float(series.mean()),
                "annualized_return": float((1.0 + series).prod() ** (config.ANNUALIZATION_DAYS / len(series)) - 1.0),
                "annualized_volatility": float(daily_std * np.sqrt(config.ANNUALIZATION_DAYS)),
                "sharpe": np.nan if daily_std == 0 else float(np.sqrt(config.ANNUALIZATION_DAYS) * series.mean() / daily_std),
                "cumulative_return": float((1.0 + series).prod() - 1.0),
            }
        )

    return daily_quantiles, pd.DataFrame(summary_rows)
