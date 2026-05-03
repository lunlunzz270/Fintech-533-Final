from __future__ import annotations

import numpy as np
import pandas as pd

import config


def cumulative_return_series(returns: pd.Series) -> pd.Series:
    """Convert daily returns to cumulative return series."""
    return (1.0 + returns.fillna(0.0)).cumprod() - 1.0


def max_drawdown(returns: pd.Series) -> float:
    """Compute maximum drawdown from a return series."""
    equity_curve = (1.0 + returns.fillna(0.0)).cumprod()
    peaks = equity_curve.cummax()
    drawdown = equity_curve / peaks - 1.0
    return float(drawdown.min()) if not drawdown.empty else np.nan


def summarize_return_series(returns: pd.Series) -> dict[str, float]:
    """Calculate performance statistics for a daily return series."""
    clean = returns.dropna()
    if clean.empty:
        return {
            "cumulative_return": np.nan,
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "sharpe_ratio": np.nan,
            "max_drawdown": np.nan,
            "hit_rate": np.nan,
            "average_daily_return": np.nan,
            "daily_return_std": np.nan,
            "skewness": np.nan,
            "kurtosis": np.nan,
            "best_day": np.nan,
            "worst_day": np.nan,
        }

    cumulative_return = float((1.0 + clean).prod() - 1.0)
    avg_daily = float(clean.mean())
    daily_std = float(clean.std(ddof=1))
    annualized_return = float((1.0 + clean).prod() ** (config.ANNUALIZATION_DAYS / len(clean)) - 1.0)
    annualized_vol = float(daily_std * np.sqrt(config.ANNUALIZATION_DAYS))
    sharpe = np.nan if daily_std == 0 else float(np.sqrt(config.ANNUALIZATION_DAYS) * avg_daily / daily_std)

    return {
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown(clean),
        "hit_rate": float((clean > 0).mean()),
        "average_daily_return": avg_daily,
        "daily_return_std": daily_std,
        "skewness": float(clean.skew()),
        "kurtosis": float(clean.kurt()),
        "best_day": float(clean.max()),
        "worst_day": float(clean.min()),
    }


def performance_summary_table(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """Build the requested gross/net performance summary table."""
    gross_metrics = summarize_return_series(daily_returns["gross_return"])
    net_metrics = summarize_return_series(daily_returns["net_return"])

    metric_rows: list[dict[str, float | str]] = []
    for metric_name in gross_metrics:
        metric_rows.append(
            {
                "metric": metric_name,
                "gross": gross_metrics[metric_name],
                "net": net_metrics[metric_name],
            }
        )

    extra_rows = [
        {
            "metric": "average_number_of_longs",
            "gross": float(daily_returns["number_of_longs"].mean()),
            "net": float(daily_returns["number_of_longs"].mean()),
        },
        {
            "metric": "average_number_of_shorts",
            "gross": float(daily_returns["number_of_shorts"].mean()),
            "net": float(daily_returns["number_of_shorts"].mean()),
        },
        {
            "metric": "average_borrow_cost",
            "gross": np.nan,
            "net": float(daily_returns["borrow_cost"].mean()),
        },
        {
            "metric": "total_borrow_cost_drag",
            "gross": np.nan,
            "net": float(daily_returns["borrow_cost"].sum()),
        },
        {
            "metric": "gross_minus_net_cumulative_difference",
            "gross": gross_metrics["cumulative_return"] - net_metrics["cumulative_return"],
            "net": np.nan,
        },
    ]
    return pd.DataFrame(metric_rows + extra_rows)


def add_drawdown_series(daily_returns: pd.DataFrame, column: str = "net_return") -> pd.DataFrame:
    """Append cumulative return and drawdown columns for plotting."""
    data = daily_returns.copy()
    data["cumulative_return"] = cumulative_return_series(data[column])
    equity_curve = 1.0 + data["cumulative_return"]
    data["drawdown"] = equity_curve / equity_curve.cummax() - 1.0
    return data
