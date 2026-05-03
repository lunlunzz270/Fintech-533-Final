from __future__ import annotations

from pathlib import Path

import pandas as pd

import config
from backtest import construct_portfolio
from data_loader import (
    download_membership_file,
    expected_ticker_ranges,
    fetch_prices_for_universe,
    filter_prices_to_membership,
    find_tickers_to_fetch,
    load_fetch_status,
    prepare_benchmark_data,
    load_membership_data,
    load_price_data,
    load_sector_data,
    save_current_ticker_snapshot,
    summarize_fetch_stats,
)
from factors import compute_factors
from ic_analysis import compute_daily_ic, summarize_ic
from metrics import performance_summary_table
from plotting import (
    plot_cumulative_returns,
    plot_drawdown,
    plot_ic_timeseries,
    plot_long_short_legs,
    plot_quantile_cumulative_returns,
)
from preprocessing import process_factors
from quantile_analysis import compute_quantile_returns
from walk_forward import run_walk_forward_backtest


def format_pct(value: float) -> str:
    """Format a decimal value as a percentage string."""
    return "nan" if pd.isna(value) else f"{value:.2%}"


def format_num(value: float) -> str:
    """Format a scalar numeric value for terminal display."""
    return "nan" if pd.isna(value) else f"{value:.3f}"


def print_key_metrics(performance_summary: pd.DataFrame) -> None:
    """Print the headline performance metrics to the terminal."""
    headline_metrics = [
        "cumulative_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "hit_rate",
    ]
    summary = performance_summary.set_index("metric")

    print("\nKey Performance Metrics")
    print("-" * 72)
    print(f"{'Metric':<24}{'Gross':>18}{'Net':>18}")
    print("-" * 72)
    for metric in headline_metrics:
        gross = summary.at[metric, "gross"] if metric in summary.index else float("nan")
        net = summary.at[metric, "net"] if metric in summary.index else float("nan")
        if metric in {"sharpe_ratio"}:
            gross_str = format_num(gross)
            net_str = format_num(net)
        else:
            gross_str = format_pct(gross)
            net_str = format_pct(net)
        print(f"{metric:<24}{gross_str:>18}{net_str:>18}")
    print("-" * 72)


def print_walk_forward_metrics(performance_summary: pd.DataFrame) -> None:
    """Print headline walk-forward strategy, benchmark, and excess metrics."""
    summary = performance_summary.pivot(index="metric", columns="series", values="value")
    headline_metrics = [
        "cumulative_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "hit_rate",
        "expected_return_per_trade",
        "average_trade_lifetime",
        "success_rate",
        "timeout_rate",
        "stop_loss_rate",
    ]

    print("\nWalk-Forward Performance Metrics")
    print("-" * 84)
    print(f"{'Metric':<24}{'Strategy':>18}{config.BENCHMARK_TICKER:>18}{'Excess':>18}")
    print("-" * 84)
    for metric in headline_metrics:
        strategy = summary.at[metric, "strategy"] if metric in summary.index and "strategy" in summary.columns else float("nan")
        benchmark = summary.at[metric, config.BENCHMARK_TICKER] if metric in summary.index and config.BENCHMARK_TICKER in summary.columns else float("nan")
        excess = summary.at[metric, "excess"] if metric in summary.index and "excess" in summary.columns else float("nan")
        
        if metric in {"sharpe_ratio", "average_trade_lifetime"}:
            strategy_str = format_num(strategy)
            benchmark_str = format_num(benchmark)
            excess_str = format_num(excess)
        else:
            strategy_str = format_pct(strategy)
            benchmark_str = format_pct(benchmark)
            excess_str = format_pct(excess)
        print(f"{metric:<24}{strategy_str:>18}{benchmark_str:>18}{excess_str:>18}")
    print("-" * 84)


def prepare_membership() -> pd.DataFrame:
    """Load or download historical S&P 500 membership data."""
    if not config.MEMBERSHIP_FILE.exists():
        download_membership_file()
    membership = load_membership_data(config.MEMBERSHIP_FILE)
    save_current_ticker_snapshot(
        membership=membership,
        as_of_date=config.END_DATE,
        output_path=config.DATA_DIR / "sp500_tickers.csv",
    )
    return membership


def prepare_prices(membership: pd.DataFrame) -> pd.DataFrame:
    """Load cached prices, optionally fetching missing coverage only when enabled."""
    existing_prices = pd.DataFrame()
    if config.PRICES_FILE.exists():
        existing_prices = load_price_data(config.PRICES_FILE)

    universe_membership = membership
    if config.STRATEGY_MODE == "walk_forward_sector_neutral":
        sectors = load_sector_data()
        universe_membership = membership[membership["ticker"].isin(sectors["ticker"])].copy()

    if not config.AUTO_FETCH_MISSING_DATA:
        if existing_prices.empty:
            raise FileNotFoundError(
                "No local price cache found at "
                f"{config.PRICES_FILE}. Set AUTO_FETCH_MISSING_DATA=True once to build it."
            )
        return filter_prices_to_membership(existing_prices, universe_membership)

    required_ranges = expected_ticker_ranges(universe_membership, config.START_DATE, config.END_DATE)
    status_cache = load_fetch_status()
    missing_tickers = find_tickers_to_fetch(
        existing_prices=existing_prices,
        required_ranges=required_ranges,
        status_cache=status_cache,
        mode=config.DATA_SOURCE_MODE.strip().lower(),
    )

    if missing_tickers:
        prices, fetch_stats, _status_cache = fetch_prices_for_universe(
            membership=universe_membership,
            start_date=config.START_DATE,
            end_date=config.END_DATE,
            output_path=config.PRICES_FILE,
            existing_prices=existing_prices,
        )
        summary = summarize_fetch_stats(fetch_stats)
        if not summary.empty:
            summary.to_csv(config.FETCH_SUMMARY_FILE, index=False)
            fetch_stats[fetch_stats["status"] == "failed"].to_csv(config.FETCH_FAILURES_FILE, index=False)
    else:
        prices = existing_prices

    prices = filter_prices_to_membership(prices, universe_membership)
    prices = prices[
        (prices["date"] >= pd.Timestamp(config.START_DATE))
        & (prices["date"] <= pd.Timestamp(config.END_DATE))
    ].copy()
    if config.STRATEGY_MODE == "walk_forward_sector_neutral":
        required_tickers = set(expected_ticker_ranges(universe_membership, config.START_DATE, config.END_DATE)["ticker"])
        available_tickers = set(prices["ticker"].dropna().unique())
        coverage = len(required_tickers & available_tickers) / len(required_tickers) if required_tickers else 1.0
        if coverage < config.MIN_SECTOR_PRICE_COVERAGE:
            missing_count = len(required_tickers - available_tickers)
            raise RuntimeError(
                "Sector universe price coverage is too low: "
                f"{coverage:.1%} available, {missing_count} tickers missing. "
                "Yahoo/yfinance may be rate-limited; wait and rerun, or restore a fuller prices.csv cache."
            )
    return prices.sort_values(["ticker", "date"]).reset_index(drop=True)


def prepare_benchmark_returns(
    start_date: str = config.START_DATE,
    end_date: str = config.END_DATE,
) -> pd.DataFrame:
    """Prepare benchmark open-to-open forward returns aligned to signal dates."""
    benchmark = prepare_benchmark_data(
        start_date=start_date,
        end_date=end_date,
        path=config.BENCHMARK_FILE,
    ).sort_values("date").reset_index(drop=True)
    benchmark["next_open"] = benchmark["open"].shift(-1)
    benchmark["next_next_open"] = benchmark["open"].shift(-2)
    benchmark["benchmark_return"] = benchmark["next_next_open"] / benchmark["next_open"] - 1.0
    return benchmark[["date", "benchmark_return"]].dropna().reset_index(drop=True)


def run_pipeline() -> None:
    """Execute the full research pipeline end-to-end."""
    config.ensure_directories()
    membership = prepare_membership()
    prices = prepare_prices(membership)
    benchmark_returns = None
    if config.STRATEGY_MODE == "walk_forward_sector_neutral":
        benchmark_returns = prepare_benchmark_returns(
            start_date=config.WALK_FORWARD_START,
            end_date=config.END_DATE,
        )
    elif config.PORTFOLIO_MODE.strip().lower() == "long_only":
        benchmark_returns = prepare_benchmark_returns()

    factor_data = compute_factors(prices)
    if config.STRATEGY_MODE == "walk_forward_sector_neutral":
        sectors = load_sector_data()
        factor_data = factor_data.merge(sectors, on="ticker", how="left")
        factor_data = process_factors(
            factor_data,
            raw_factor_columns=config.RAW_FACTOR_COLUMNS,
            group_column=config.NEUTRALIZATION_GROUP,
        )
        factor_data.to_csv(config.FACTOR_DATA_FILE, index=False)

        if benchmark_returns is None:
            benchmark_returns = prepare_benchmark_returns()
        (
            daily_returns,
            blotter,
            ledger,
            performance_summary,
            ic_summary,
            fold_summary,
            factor_weights,
        ) = run_walk_forward_backtest(factor_data, benchmark_returns)

        daily_returns.to_csv(config.DAILY_RETURNS_FILE, index=False)
        blotter.to_csv(config.BLOTTER_FILE, index=False)
        ledger.to_csv(config.LEDGER_FILE, index=False)
        performance_summary.to_csv(config.PERFORMANCE_SUMMARY_FILE, index=False)
        ic_summary.to_csv(config.IC_SUMMARY_FILE, index=False)
        fold_summary.to_csv(config.WALK_FORWARD_FOLDS_FILE, index=False)
        factor_weights.to_csv(config.WALK_FORWARD_WEIGHTS_FILE, index=False)
        print_walk_forward_metrics(performance_summary)

        if not daily_returns.empty:
            plot_cumulative_returns(daily_returns, benchmark_returns=benchmark_returns)
            plot_drawdown(daily_returns)
            plot_long_short_legs(daily_returns, benchmark_returns=benchmark_returns)
        return

    factor_data = process_factors(factor_data)
    factor_data.to_csv(config.FACTOR_DATA_FILE, index=False)

    daily_returns, _positions = construct_portfolio(factor_data)
    daily_returns.to_csv(config.DAILY_RETURNS_FILE, index=False)

    performance_summary = performance_summary_table(daily_returns)
    performance_summary.to_csv(config.PERFORMANCE_SUMMARY_FILE, index=False)
    print_key_metrics(performance_summary)

    daily_ic = compute_daily_ic(factor_data)
    ic_summary = summarize_ic(daily_ic)
    ic_summary.to_csv(config.IC_SUMMARY_FILE, index=False)

    quantile_daily, quantile_summary = compute_quantile_returns(factor_data)
    quantile_summary.to_csv(config.QUANTILE_SUMMARY_FILE, index=False)

    if not daily_returns.empty:
        plot_cumulative_returns(daily_returns, benchmark_returns=benchmark_returns)
        plot_drawdown(daily_returns)
        plot_long_short_legs(daily_returns, benchmark_returns=benchmark_returns)

    if not daily_ic.empty and "combined_score_ic" in daily_ic.columns:
        plot_ic_timeseries(daily_ic)

    if not quantile_daily.empty:
        plot_quantile_cumulative_returns(quantile_daily)


if __name__ == "__main__":
    run_pipeline()
