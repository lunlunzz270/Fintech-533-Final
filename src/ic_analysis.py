from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config


def spearman_ic(series_x: pd.Series, series_y: pd.Series) -> float:
    """Safely compute a Spearman rank correlation."""
    aligned = pd.concat([series_x, series_y], axis=1).dropna()
    if len(aligned) < 3:
        return np.nan
    if aligned.iloc[:, 0].nunique() < 2 or aligned.iloc[:, 1].nunique() < 2:
        return np.nan
    correlation, _ = spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
    return float(correlation) if correlation is not None else np.nan


def compute_daily_ic(
    factor_data: pd.DataFrame,
    factor_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Compute daily Spearman Rank IC series."""
    factor_columns = factor_columns or (config.Z_FACTOR_COLUMNS + ["combined_score"])
    rows: list[dict[str, float | str]] = []

    for date, daily in factor_data.groupby("date", sort=True):
        row: dict[str, float | str] = {"date": date}
        for factor in factor_columns:
            row[factor.replace("_z", "") + "_ic" if factor.endswith("_z") else f"{factor}_ic"] = spearman_ic(
                daily[factor],
                daily["forward_return"],
            )
        rows.append(row)

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def summarize_ic(daily_ic: pd.DataFrame) -> pd.DataFrame:
    """Summarize IC series with mean, t-stat, and ICIR."""
    rows: list[dict[str, float | str]] = []
    for column in daily_ic.columns:
        if column == "date":
            continue
        series = daily_ic[column].dropna()
        n_obs = len(series)
        if n_obs == 0:
            rows.append(
                {
                    "factor": column,
                    "mean_ic": np.nan,
                    "ic_std": np.nan,
                    "ic_t_stat": np.nan,
                    "icir": np.nan,
                    "annualized_icir": np.nan,
                    "positive_ic_ratio": np.nan,
                    "n_obs": 0,
                }
            )
            continue
        ic_std = float(series.std(ddof=1)) if n_obs > 1 else np.nan
        mean_ic = float(series.mean())
        ic_t_stat = np.nan if not ic_std or np.isnan(ic_std) else mean_ic / (ic_std / np.sqrt(n_obs))
        icir = np.nan if not ic_std or np.isnan(ic_std) else mean_ic / ic_std
        rows.append(
            {
                "factor": column,
                "mean_ic": mean_ic,
                "ic_std": ic_std,
                "ic_t_stat": ic_t_stat,
                "icir": icir,
                "annualized_icir": np.nan if pd.isna(icir) else icir * np.sqrt(config.ANNUALIZATION_DAYS),
                "positive_ic_ratio": float((series > 0).mean()),
                "n_obs": n_obs,
            }
        )
    return pd.DataFrame(rows)
