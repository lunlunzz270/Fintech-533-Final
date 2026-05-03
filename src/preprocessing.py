from __future__ import annotations

import numpy as np
import pandas as pd

import config


def winsorize_series(series: pd.Series, lower: float, upper: float) -> pd.Series:
    """Winsorize a cross-sectional series at the specified quantiles."""
    valid = series.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index)
    lower_bound = valid.quantile(lower)
    upper_bound = valid.quantile(upper)
    return series.clip(lower=lower_bound, upper=upper_bound)


def zscore_series(series: pd.Series) -> pd.Series:
    """Compute a cross-sectional z-score."""
    valid = series.dropna()
    if len(valid) < 2:
        return pd.Series(np.nan, index=series.index)
    std = valid.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.nan, index=series.index)
    mean = valid.mean()
    return (series - mean) / std


def process_factors(
    factor_data: pd.DataFrame,
    raw_factor_columns: list[str] | None = None,
    group_column: str | None = None,
) -> pd.DataFrame:
    """Winsorize and z-score factors cross-sectionally by date or date/group."""
    raw_factor_columns = raw_factor_columns or config.RAW_FACTOR_COLUMNS
    data = factor_data.copy().sort_values(["date", "ticker"]).reset_index(drop=True)
    group_keys = ["date"]
    if group_column and group_column in data.columns:
        data = data.dropna(subset=[group_column]).copy()
        group_keys = ["date", group_column]

    for column in raw_factor_columns:
        winsorized_column = f"{column}_winsorized"
        z_column = f"{column}_z"
        data[winsorized_column] = data.groupby(group_keys, dropna=False)[column].transform(
            lambda series: winsorize_series(series, config.WINSOR_LOWER, config.WINSOR_UPPER)
        )
        data[z_column] = data.groupby(group_keys, dropna=False)[winsorized_column].transform(zscore_series)

    z_columns = [f"{column}_z" for column in raw_factor_columns]
    valid_factor_counts = data[z_columns].notna().sum(axis=1)
    data["valid_factor_count"] = valid_factor_counts
    data["combined_score"] = data[z_columns].mean(axis=1, skipna=True)
    data.loc[data["valid_factor_count"] < config.MIN_VALID_FACTORS, "combined_score"] = np.nan
    return data
