from __future__ import annotations

import pandas as pd

import config
from metrics import cumulative_return_series


def plot_cumulative_returns(
    daily_returns: pd.DataFrame,
    benchmark_returns: pd.DataFrame | None = None,
    output_path: str | None = None,
) -> None:
    """Plot cumulative returns with portfolio-mode-aware labeling."""
    import matplotlib.pyplot as plt

    output_path = output_path or str(config.CUMULATIVE_RETURNS_PLOT)
    frame = daily_returns.copy()

    plt.figure(figsize=(12, 6))
    if config.PORTFOLIO_MODE.strip().lower() == "long_only":
        frame["strategy_cumulative"] = cumulative_return_series(frame["net_return"])
        plt.plot(frame["date"], frame["strategy_cumulative"], label="Long-Only Strategy")
        if benchmark_returns is not None and not benchmark_returns.empty:
            benchmark_frame = benchmark_returns.copy()
            benchmark_frame = frame[["date"]].merge(benchmark_frame, on="date", how="left")
            benchmark_frame["benchmark_cumulative"] = cumulative_return_series(benchmark_frame["benchmark_return"])
            plt.plot(benchmark_frame["date"], benchmark_frame["benchmark_cumulative"], label=f"{config.BENCHMARK_TICKER} Benchmark")
        plt.title("Long-Only Strategy vs Benchmark")
    else:
        frame["gross_cumulative"] = cumulative_return_series(frame["gross_return"])
        frame["net_cumulative"] = cumulative_return_series(frame["net_return"])
        plt.plot(frame["date"], frame["gross_cumulative"], label="Gross")
        plt.plot(frame["date"], frame["net_cumulative"], label="Net")
        plt.title("Cumulative Returns")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_drawdown(daily_returns: pd.DataFrame, output_path: str | None = None) -> None:
    """Plot net-return drawdown."""
    import matplotlib.pyplot as plt

    output_path = output_path or str(config.DRAWDOWN_PLOT)
    cumulative = (1.0 + daily_returns["net_return"].fillna(0.0)).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1.0

    plt.figure(figsize=(12, 6))
    plt.plot(daily_returns["date"], drawdown, color="firebrick")
    plt.title("Net Strategy Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_ic_timeseries(daily_ic: pd.DataFrame, output_path: str | None = None) -> None:
    """Plot daily combined-score IC and its rolling mean."""
    import matplotlib.pyplot as plt

    output_path = output_path or str(config.IC_TIMESERIES_PLOT)
    column = "combined_score_ic"
    frame = daily_ic[["date", column]].copy()
    frame["rolling_mean"] = frame[column].rolling(config.ROLLING_IC_WINDOW).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(frame["date"], frame[column], label="Daily Rank IC", alpha=0.6)
    plt.plot(frame["date"], frame["rolling_mean"], label=f"Rolling {config.ROLLING_IC_WINDOW}D Mean", linewidth=2)
    plt.axhline(0.0, color="black", linewidth=1, linestyle="--")
    plt.title("Combined Score Rank IC")
    plt.xlabel("Date")
    plt.ylabel("Rank IC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_quantile_cumulative_returns(daily_quantiles: pd.DataFrame, output_path: str | None = None) -> None:
    """Plot cumulative returns for Q1 through Q5."""
    import matplotlib.pyplot as plt

    output_path = output_path or str(config.QUANTILE_PLOT)
    plt.figure(figsize=(12, 6))
    for quantile in range(1, config.NUM_QUANTILES + 1):
        column = f"Q{quantile}"
        if column in daily_quantiles:
            plt.plot(daily_quantiles["date"], cumulative_return_series(daily_quantiles[column]), label=column)
    plt.title("Quantile Cumulative Returns")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_long_short_legs(
    daily_returns: pd.DataFrame,
    benchmark_returns: pd.DataFrame | None = None,
    output_path: str | None = None,
) -> None:
    """Plot cumulative long/short legs or long-only strategy vs benchmark."""
    import matplotlib.pyplot as plt

    output_path = output_path or str(config.LONG_SHORT_LEGS_PLOT)
    frame = daily_returns.copy()

    plt.figure(figsize=(12, 6))
    if config.PORTFOLIO_MODE.strip().lower() == "long_only":
        frame["strategy_cumulative"] = cumulative_return_series(frame["long_leg_return"])
        plt.plot(frame["date"], frame["strategy_cumulative"], label="Long-Only Strategy")
        if benchmark_returns is not None and not benchmark_returns.empty:
            benchmark_frame = benchmark_returns.copy()
            benchmark_frame = frame[["date"]].merge(benchmark_frame, on="date", how="left")
            benchmark_frame["benchmark_cumulative"] = cumulative_return_series(benchmark_frame["benchmark_return"])
            plt.plot(benchmark_frame["date"], benchmark_frame["benchmark_cumulative"], label=f"{config.BENCHMARK_TICKER} Benchmark")
        plt.title("Long-Only Strategy vs Benchmark")
    else:
        frame["long_cumulative"] = cumulative_return_series(frame["long_leg_return"])
        frame["short_cumulative"] = cumulative_return_series(-frame["short_leg_return"])
        frame["spread_cumulative"] = cumulative_return_series(frame["long_short_spread"])
        plt.plot(frame["date"], frame["long_cumulative"], label="Long Leg")
        plt.plot(frame["date"], frame["short_cumulative"], label="Short Leg")
        plt.plot(frame["date"], frame["spread_cumulative"], label="Long-Short Spread")
        plt.title("Long/Short Leg Performance")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
