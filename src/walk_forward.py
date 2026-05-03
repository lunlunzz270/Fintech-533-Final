from __future__ import annotations

import math

import numpy as np
import pandas as pd

import config
from ic_analysis import compute_daily_ic, summarize_ic
from metrics import summarize_return_series, summarize_trade_stats


def build_test_windows() -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Build rolling 6-month out-of-sample windows through the configured end date."""
    start = pd.Timestamp(config.WALK_FORWARD_START)
    end = pd.Timestamp(config.END_DATE)
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    test_start = start
    while test_start <= end:
        test_end = min(
            test_start + pd.DateOffset(months=config.WALK_FORWARD_TEST_MONTHS) - pd.DateOffset(days=1),
            end,
        )
        windows.append((test_start, test_end))
        test_start = test_start + pd.DateOffset(months=config.WALK_FORWARD_TEST_MONTHS)
    return windows


def compute_factor_weights(
    daily_ic: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
) -> tuple[dict[str, float], dict[str, float]]:
    """Use positive trailing train-period IC means as normalized factor weights."""
    train_ic = daily_ic[(daily_ic["date"] >= train_start) & (daily_ic["date"] <= train_end)]
    mean_ics: dict[str, float] = {}
    positive_ics: list[float] = []

    for factor in config.RAW_FACTOR_COLUMNS:
        column = f"{factor}_ic"
        mean_ic = float(train_ic[column].dropna().mean()) if column in train_ic else np.nan
        mean_ics[factor] = mean_ic
        positive_ics.append(max(0.0, mean_ic) if np.isfinite(mean_ic) else 0.0)

    total = float(np.sum(positive_ics))
    if total <= 0.0:
        equal_weight = 1.0 / len(config.RAW_FACTOR_COLUMNS)
        weights = {factor: equal_weight for factor in config.RAW_FACTOR_COLUMNS}
    else:
        weights = {
            factor: positive_ic / total
            for factor, positive_ic in zip(config.RAW_FACTOR_COLUMNS, positive_ics)
        }
    return weights, mean_ics


def add_weighted_score(factor_data: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """Create a combined score from fixed train-period factor weights."""
    data = factor_data.copy()
    score = np.zeros(len(data), dtype=float)
    has_signal = np.zeros(len(data), dtype=bool)

    for factor, weight in weights.items():
        column = f"{factor}_z"
        if column not in data:
            continue
        values = pd.to_numeric(data[column], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(values)
        score[valid] += float(weight) * values[valid]
        has_signal |= valid

    data["combined_score"] = np.where(has_signal, score, np.nan)
    return data


def construct_sector_neutral_weekly_portfolio(
    scored_data: pd.DataFrame,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select top-scoring names inside each sector every configured weekly step."""
    data = scored_data[
        (scored_data["date"] >= test_start) & (scored_data["date"] <= test_end)
    ].copy()
    if data.empty:
        return empty_daily_returns(), empty_blotter()

    rebalance_dates = pd.Series(sorted(pd.to_datetime(data["date"].dropna().unique())))
    rebalance_dates = set(rebalance_dates.iloc[:: config.WEEKLY_REBALANCE_DAYS])
    rows: list[dict[str, float | int | str | pd.Timestamp]] = []
    blotter_rows: list[dict[str, float | int | str | bool | pd.Timestamp]] = []
    trade_id = 1

    for date, daily in data.groupby("date", sort=True):
        date = pd.Timestamp(date)
        if date not in rebalance_dates:
            continue

        eligible = daily.dropna(subset=["combined_score", "forward_return", "sector"]).copy()
        if len(eligible) < config.MIN_CROSS_SECTION_SIZE:
            continue

        long_groups: list[pd.DataFrame] = []
        for sector, sector_frame in eligible.groupby("sector"):
            if len(sector_frame) < 8:
                continue
            n_select = max(1, math.floor(len(sector_frame) * config.PORTFOLIO_SELECTION_FRACTION))
            selected = sector_frame.sort_values(["combined_score", "ticker"]).tail(n_select).copy()
            selected["sector"] = sector
            long_groups.append(selected)

        if not long_groups:
            continue

        longs = pd.concat(long_groups, ignore_index=True)
        longs["raw_forward_return"] = longs["forward_return"]
        longs["stop_hit"] = False
        if config.ENABLE_STOP_LOSS and {"entry_open", "hold_day_low"}.issubset(longs.columns):
            stop_hit = longs["hold_day_low"] <= longs["entry_open"] * (1.0 - config.STOP_LOSS_PCT)
            longs.loc[stop_hit, "stop_hit"] = True
            longs.loc[stop_hit, "forward_return"] = -config.STOP_LOSS_PCT

        # Assign trade fates: success, timeout, or stop-loss
        longs["fate"] = "timeout"
        longs.loc[longs["forward_return"] > 0, "fate"] = "success"
        longs.loc[longs["stop_hit"], "fate"] = "stop-loss"
        longs["trade_lifetime"] = 1.0

        net_return = float(longs["forward_return"].mean())
        exit_open = longs["entry_open"] * (1.0 + longs["forward_return"])
        weight = 1.0 / len(longs)
        for selected, exit_price in zip(longs.itertuples(index=False), exit_open):
            blotter_rows.append(
                {
                    "trade_id": trade_id,
                    "signal_date": date,
                    "ticker": selected.ticker,
                    "sector": selected.sector,
                    "side": "long",
                    "entry_date": date + pd.offsets.BDay(1),
                    "exit_date": date + pd.offsets.BDay(2),
                    "entry_price": selected.entry_open,
                    "exit_price": float(exit_price),
                    "weight": weight,
                    "combined_score": selected.combined_score,
                    "raw_forward_return": selected.raw_forward_return,
                    "return_pct": selected.forward_return,
                    "stop_hit": bool(selected.stop_hit),
                    "fate": selected.fate,
                    "trade_lifetime": selected.trade_lifetime,
                    "fold": f"{test_start.date()} to {test_end.date()}",
                }
            )
            trade_id += 1
        rows.append(
            {
                "date": date,
                "gross_return": net_return,
                "borrow_cost": 0.0,
                "net_return": net_return,
                "long_leg_return": net_return,
                "short_leg_return": 0.0,
                "long_short_spread": net_return,
                "number_of_longs": int(len(longs)),
                "number_of_shorts": 0,
                "number_of_stop_hits": int(longs["stop_hit"].sum()),
                "average_long_score": float(longs["combined_score"].mean()),
                "average_short_score": np.nan,
            }
        )

    daily_returns = (
        pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        if rows
        else empty_daily_returns()
    )
    blotter = (
        pd.DataFrame(blotter_rows).sort_values(["signal_date", "ticker"]).reset_index(drop=True)
        if blotter_rows
        else empty_blotter()
    )
    return daily_returns, blotter


def empty_daily_returns() -> pd.DataFrame:
    """Return an empty daily-return frame matching the standard output schema."""
    return pd.DataFrame(
        columns=[
            "date",
            "gross_return",
            "borrow_cost",
            "net_return",
            "long_leg_return",
            "short_leg_return",
            "long_short_spread",
            "number_of_longs",
            "number_of_shorts",
            "number_of_stop_hits",
            "average_long_score",
            "average_short_score",
        ]
    )


def empty_blotter() -> pd.DataFrame:
    """Return an empty selected-position blotter."""
    return pd.DataFrame(
        columns=[
            "trade_id",
            "signal_date",
            "ticker",
            "sector",
            "side",
            "entry_date",
            "exit_date",
            "entry_price",
            "exit_price",
            "weight",
            "combined_score",
            "raw_forward_return",
            "return_pct",
            "stop_hit",
            "fate",
            "trade_lifetime",
            "fold",
        ]
    )


def summarize_strategy_vs_benchmark(
    daily_returns: pd.DataFrame,
    benchmark_returns: pd.DataFrame,
    blotter: pd.DataFrame = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Append benchmark/excess returns and summarize stitched OOS performance."""
    if daily_returns.empty:
        merged = daily_returns.copy()
        merged["benchmark_return"] = np.nan
        merged["excess_return"] = np.nan
    else:
        merged = daily_returns.merge(benchmark_returns, on="date", how="left")
        merged["excess_return"] = merged["net_return"] - merged["benchmark_return"]

    rows: list[dict[str, float | str]] = []
    for label, column in [
        ("strategy", "net_return"),
        (config.BENCHMARK_TICKER, "benchmark_return"),
        ("excess", "excess_return"),
    ]:
        metrics = summarize_return_series(merged[column])
        for metric, value in metrics.items():
            rows.append({"series": label, "metric": metric, "value": value})

    if blotter is not None:
        trade_stats = summarize_trade_stats(blotter)
        for metric, value in trade_stats.items():
            rows.append({"series": "strategy", "metric": metric, "value": value})

    return merged, pd.DataFrame(rows)


def build_ledger(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """Build a daily aggregate ledger from stitched walk-forward returns."""
    ledger = daily_returns.copy().sort_values("date").reset_index(drop=True)
    if ledger.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "fold",
                "portfolio_value",
                "daily_pnl",
                "daily_return",
                "benchmark_return",
                "excess_return",
                "number_of_longs",
                "number_of_stop_hits",
                "average_long_score",
            ]
        )

    ledger["portfolio_value"] = config.INITIAL_CAPITAL * (1.0 + ledger["net_return"]).cumprod()
    previous_value = ledger["portfolio_value"].shift(1).fillna(config.INITIAL_CAPITAL)
    ledger["daily_pnl"] = ledger["portfolio_value"] - previous_value
    return ledger[
        [
            "date",
            "fold",
            "portfolio_value",
            "daily_pnl",
            "net_return",
            "benchmark_return",
            "excess_return",
            "number_of_longs",
            "number_of_stop_hits",
            "average_long_score",
        ]
    ].rename(columns={"net_return": "daily_return"})


def run_walk_forward_backtest(
    factor_data: pd.DataFrame,
    benchmark_returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the configured sector-neutral walk-forward strategy."""
    daily_ic = compute_daily_ic(factor_data)
    ic_summary = summarize_ic(daily_ic)
    daily_pieces: list[pd.DataFrame] = []
    blotter_pieces: list[pd.DataFrame] = []
    fold_rows: list[dict[str, float | int | str]] = []
    weight_rows: list[dict[str, float | str]] = []

    for test_start, test_end in build_test_windows():
        train_end = test_start - pd.DateOffset(days=1)
        train_start = test_start - pd.DateOffset(years=config.WALK_FORWARD_TRAIN_YEARS)
        weights, mean_ics = compute_factor_weights(daily_ic, train_start, train_end)

        scored = add_weighted_score(factor_data, weights)
        fold_daily, fold_blotter = construct_sector_neutral_weekly_portfolio(scored, test_start, test_end)
        if not fold_daily.empty:
            fold_daily["fold"] = f"{test_start.date()} to {test_end.date()}"
            daily_pieces.append(fold_daily)
        if not fold_blotter.empty:
            blotter_pieces.append(fold_blotter)

        fold_with_benchmark, fold_summary = summarize_strategy_vs_benchmark(
            fold_daily,
            benchmark_returns,
            blotter=fold_blotter,
        )
        summary_lookup = fold_summary.pivot(index="metric", columns="series", values="value")
        fold_rows.append(
            {
                "fold": f"{test_start.date()} to {test_end.date()}",
                "train_start": str(train_start.date()),
                "train_end": str(train_end.date()),
                "days": int(len(fold_with_benchmark)),
                "strategy_cumulative_return": lookup(summary_lookup, "cumulative_return", "strategy"),
                "benchmark_cumulative_return": lookup(summary_lookup, "cumulative_return", config.BENCHMARK_TICKER),
                "excess_cumulative_return": lookup(summary_lookup, "cumulative_return", "excess"),
                "strategy_sharpe": lookup(summary_lookup, "sharpe_ratio", "strategy"),
                "excess_sharpe": lookup(summary_lookup, "sharpe_ratio", "excess"),
                "average_number_of_longs": float(fold_daily["number_of_longs"].mean()) if not fold_daily.empty else np.nan,
            }
        )

        weight_row: dict[str, float | str] = {
            "fold": f"{test_start.date()} to {test_end.date()}",
            "train_start": str(train_start.date()),
            "train_end": str(train_end.date()),
        }
        for factor in config.RAW_FACTOR_COLUMNS:
            weight_row[f"weight_{factor}"] = weights[factor]
            weight_row[f"train_mean_ic_{factor}"] = mean_ics[factor]
        weight_rows.append(weight_row)

    daily_returns = (
        pd.concat(daily_pieces, ignore_index=True).sort_values("date").reset_index(drop=True)
        if daily_pieces
        else empty_daily_returns()
    )
    blotter = (
        pd.concat(blotter_pieces, ignore_index=True).sort_values(["signal_date", "ticker"]).reset_index(drop=True)
        if blotter_pieces
        else empty_blotter()
    )
    daily_returns, performance_summary = summarize_strategy_vs_benchmark(
        daily_returns,
        benchmark_returns,
        blotter=blotter,
    )
    ledger = build_ledger(daily_returns)
    return (
        daily_returns,
        blotter,
        ledger,
        performance_summary,
        ic_summary,
        pd.DataFrame(fold_rows),
        pd.DataFrame(weight_rows),
    )


def lookup(table: pd.DataFrame, metric: str, series: str) -> float:
    """Safely pull one metric out of a pivoted summary table."""
    if metric not in table.index or series not in table.columns:
        return np.nan
    return float(table.at[metric, series])
