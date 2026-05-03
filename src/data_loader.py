from __future__ import annotations

from dataclasses import dataclass
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import config

try:
    import shinybroker as sb
except ImportError:  # pragma: no cover - optional dependency at import time
    sb = None


PRICE_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]


@dataclass(frozen=True)
class FetchStats:
    ticker: str
    source: str
    rows: int
    status: str


def expected_ticker_ranges(
    membership: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Return expected date coverage by ticker within the requested backtest range."""
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    overlap = membership[
        (membership["start_date"] <= end_ts)
        & (membership["end_date"].isna() | (membership["end_date"] > start_ts))
    ].copy()
    if overlap.empty:
        return pd.DataFrame(columns=["ticker", "required_start", "required_end"])

    overlap["required_start"] = overlap["start_date"].clip(lower=start_ts)
    effective_end = overlap["end_date"].fillna(end_ts + pd.Timedelta(days=1)) - pd.Timedelta(days=1)
    overlap["required_end"] = effective_end.clip(upper=end_ts)

    overlap = overlap[overlap["required_start"] <= overlap["required_end"]]
    return (
        overlap.groupby("ticker", as_index=False)
        .agg(required_start=("required_start", "min"), required_end=("required_end", "max"))
        .sort_values("ticker")
        .reset_index(drop=True)
    )


def download_membership_file(
    target_path: Path = config.MEMBERSHIP_FILE,
    source_url: str = config.MEMBERSHIP_SOURCE_URL,
) -> Path:
    """Download historical S&P 500 membership data from GitHub."""
    response = requests.get(source_url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(response.content)
    return target_path


def download_sector_file(
    target_path: Path = config.SECTOR_FILE,
    source_url: str = config.SECTOR_SOURCE_URL,
) -> Path:
    """Download current S&P 500 sector classifications."""
    response = requests.get(source_url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(response.content)
    return target_path


def load_sector_data(path: Path = config.SECTOR_FILE) -> pd.DataFrame:
    """Load ticker-to-sector data from DataHub/Wikipedia style constituents CSV."""
    if not path.exists():
        download_sector_file(path)

    sectors = pd.read_csv(path)
    symbol_column = "Symbol" if "Symbol" in sectors.columns else "ticker"
    sector_column = "GICS Sector" if "GICS Sector" in sectors.columns else "sector"
    if symbol_column not in sectors.columns or sector_column not in sectors.columns:
        raise ValueError(
            "Sector file must include Symbol/ticker and GICS Sector/sector columns."
        )

    result = sectors[[symbol_column, sector_column]].copy()
    result = result.rename(columns={symbol_column: "ticker", sector_column: "sector"})
    result["ticker"] = result["ticker"].astype(str).str.strip().str.replace(".", "-", regex=False)
    result["sector"] = result["sector"].astype(str).str.strip()
    result = result.dropna(subset=["ticker", "sector"])
    return result.drop_duplicates("ticker").sort_values("ticker").reset_index(drop=True)


def load_membership_data(path: Path = config.MEMBERSHIP_FILE) -> pd.DataFrame:
    """Load membership data and normalize it to ticker/start/end intervals."""
    if not path.exists():
        raise FileNotFoundError(f"Membership file not found: {path}")

    membership = pd.read_csv(path)
    membership.columns = [column.strip().lower() for column in membership.columns]

    if {"ticker", "start_date", "end_date"}.issubset(membership.columns):
        intervals = membership[["ticker", "start_date", "end_date"]].copy()
        intervals["ticker"] = intervals["ticker"].astype(str).str.strip()
        intervals["start_date"] = pd.to_datetime(intervals["start_date"])
        intervals["end_date"] = pd.to_datetime(intervals["end_date"], errors="coerce")
        intervals = intervals.dropna(subset=["ticker", "start_date"])
        return intervals.sort_values(["ticker", "start_date"]).reset_index(drop=True)

    if {"date", "tickers"}.issubset(membership.columns):
        return _membership_list_to_intervals(membership)

    raise ValueError(
        "Membership file must contain either "
        "['ticker', 'start_date', 'end_date'] or ['date', 'tickers'] columns."
    )


def _membership_list_to_intervals(membership: pd.DataFrame) -> pd.DataFrame:
    membership = membership[["date", "tickers"]].copy()
    membership["date"] = pd.to_datetime(membership["date"])
    membership["tickers"] = membership["tickers"].fillna("")

    exploded_rows: list[dict[str, object]] = []
    for row in membership.itertuples(index=False):
        date = pd.Timestamp(row.date)
        tickers = [ticker.strip() for ticker in str(row.tickers).split(",") if ticker.strip()]
        exploded_rows.extend({"date": date, "ticker": ticker} for ticker in tickers)

    exploded = pd.DataFrame(exploded_rows)
    if exploded.empty:
        return pd.DataFrame(columns=["ticker", "start_date", "end_date"])

    exploded = exploded.sort_values(["ticker", "date"]).reset_index(drop=True)
    exploded["prev_date"] = exploded.groupby("ticker")["date"].shift(1)
    exploded["new_interval"] = (
        exploded["prev_date"].isna()
        | ((exploded["date"] - exploded["prev_date"]).dt.days > 7)
    )
    exploded["interval_id"] = exploded.groupby("ticker")["new_interval"].cumsum()

    intervals = (
        exploded.groupby(["ticker", "interval_id"], as_index=False)
        .agg(start_date=("date", "min"), last_seen_date=("date", "max"))
        .drop(columns=["interval_id"])
    )
    intervals["end_date"] = intervals["last_seen_date"] + pd.Timedelta(days=1)
    return intervals.drop(columns=["last_seen_date"]).sort_values(["ticker", "start_date"]).reset_index(drop=True)


def tickers_for_date_range(
    membership: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> list[str]:
    """Return tickers whose membership interval overlaps the requested date range."""
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    overlap = membership[
        (membership["start_date"] <= end_ts)
        & (membership["end_date"].isna() | (membership["end_date"] > start_ts))
    ]
    return sorted(overlap["ticker"].dropna().unique().tolist())


def build_contract(symbol: str) -> "sb.Contract":
    if sb is None:
        raise ImportError("shinybroker is not installed in the active Python environment.")
    return sb.Contract(
        {
            "symbol": symbol,
            "secType": "STK",
            "exchange": "SMART",
            "currency": "USD",
        }
    )


def ibkr_duration_string(start_date: str, end_date: str) -> str:
    """Convert a date range to an IBKR duration string."""
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    years = max(1, int(np.ceil((end_ts - start_ts).days / 365.25)))
    return f"{years} Y"


def fetch_history_ibkr(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily OHLCV from IBKR via shinybroker."""
    if sb is None:
        return pd.DataFrame()

    response = sb.fetch_historical_data(
        contract=build_contract(symbol),
        durationStr=ibkr_duration_string(start_date, end_date),
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=config.IBKR_USE_RTH,
        host=config.IBKR_HOST,
        port=config.IBKR_PORT,
        client_id=config.IBKR_CLIENT_ID,
        timeout=config.IBKR_TIMEOUT,
    )

    if not isinstance(response, dict) or "hst_dta" not in response:
        return pd.DataFrame()

    history = response["hst_dta"].copy()
    if history.empty:
        return pd.DataFrame()

    history["date"] = pd.to_datetime(history["timestamp"]).dt.normalize()
    history = history.rename(columns=str.lower)
    history = history[["date", "open", "high", "low", "close", "volume"]].copy()
    history["ticker"] = symbol
    history = history[(history["date"] >= pd.Timestamp(start_date)) & (history["date"] <= pd.Timestamp(end_date))]
    return history.reset_index(drop=True)


def yfinance_symbol(symbol: str) -> str:
    """Translate dot tickers to Yahoo's dash convention."""
    return symbol.replace(".", "-")


def fetch_history_yfinance(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily adjusted OHLCV from yfinance."""
    import yfinance as yf

    if config.QUIET_YFINANCE_OUTPUT:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            history = yf.download(
                tickers=yfinance_symbol(symbol),
                start=start_date,
                end=(pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                interval="1d",
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=config.YFINANCE_THREADS,
            )
    else:
        history = yf.download(
            tickers=yfinance_symbol(symbol),
            start=start_date,
            end=(pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=False,
            actions=False,
            progress=False,
            threads=config.YFINANCE_THREADS,
        )
    if history.empty:
        return pd.DataFrame(columns=PRICE_COLUMNS)

    if isinstance(history.columns, pd.MultiIndex):
        history.columns = history.columns.get_level_values(0)

    history = history.reset_index().rename(columns=str.lower)
    rename_map = {"adj close": "adj_close"}
    history = history.rename(columns=rename_map)

    required = {"date", "open", "high", "low", "close", "adj_close", "volume"}
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"Missing Yahoo columns for {symbol}: {sorted(missing)}")

    history["date"] = pd.to_datetime(history["date"]).dt.normalize()
    history["ticker"] = symbol
    return history[PRICE_COLUMNS].dropna(subset=["date", "open", "high", "low", "close"]).reset_index(drop=True)


def load_benchmark_data(path: Path = config.BENCHMARK_FILE) -> pd.DataFrame:
    """Load cached benchmark OHLCV data from CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")
    benchmark = pd.read_csv(path)
    return clean_price_data(benchmark)


def fetch_benchmark_history(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch benchmark data with Yahoo first and IBKR fallback."""
    try:
        benchmark = fetch_history_yfinance(config.BENCHMARK_TICKER, start_date, end_date)
    except Exception:
        benchmark = pd.DataFrame(columns=PRICE_COLUMNS)
    if not benchmark.empty:
        return benchmark

    try:
        ibkr_history = fetch_history_ibkr(config.BENCHMARK_TICKER, start_date, end_date)
    except Exception:
        ibkr_history = pd.DataFrame(columns=PRICE_COLUMNS)
    if ibkr_history.empty:
        return pd.DataFrame(columns=PRICE_COLUMNS)

    ibkr_history["adj_close"] = ibkr_history["close"]
    return ibkr_history[PRICE_COLUMNS].copy()


def prepare_benchmark_data(
    start_date: str = config.START_DATE,
    end_date: str = config.END_DATE,
    path: Path = config.BENCHMARK_FILE,
) -> pd.DataFrame:
    """Load cached benchmark data and extend it when config requires a wider date range."""
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    if path.exists():
        benchmark = load_benchmark_data(path)
    else:
        benchmark = pd.DataFrame(columns=PRICE_COLUMNS)

    has_coverage = (
        not benchmark.empty
        and benchmark["date"].min() <= start_ts
        and benchmark["date"].max() >= end_ts
    )

    if not has_coverage:
        if not config.AUTO_FETCH_MISSING_DATA:
            raise FileNotFoundError(
                "Benchmark cache does not cover the requested date range at "
                f"{path}. Set AUTO_FETCH_MISSING_DATA=True once to build or extend it."
            )
        refreshed = fetch_benchmark_history(start_date, end_date)
        if refreshed.empty:
            raise RuntimeError(
                f"Unable to fetch benchmark data for {config.BENCHMARK_TICKER} "
                f"over {start_date} to {end_date}. Existing cache ends at "
                f"{benchmark['date'].max().date() if not benchmark.empty else 'N/A'}."
            )
        benchmark = merge_price_frames(benchmark, refreshed)
        path.parent.mkdir(parents=True, exist_ok=True)
        benchmark.to_csv(path, index=False)

    benchmark = benchmark[
        (benchmark["date"] >= start_ts) & (benchmark["date"] <= end_ts)
    ].copy()
    return clean_price_data(benchmark)


def fetch_histories_yfinance_batch(
    symbols: list[str],
    start_date: str,
    end_date: str,
) -> dict[str, pd.DataFrame]:
    """Fetch multiple tickers at once from yfinance and split them back into per-ticker frames."""
    import yfinance as yf

    if not symbols:
        return {}

    yahoo_to_original = {yfinance_symbol(symbol): symbol for symbol in symbols}
    if config.QUIET_YFINANCE_OUTPUT:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            downloaded = yf.download(
                tickers=list(yahoo_to_original.keys()),
                start=start_date,
                end=(pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                interval="1d",
                auto_adjust=False,
                actions=False,
                progress=False,
                group_by="ticker",
                threads=config.YFINANCE_THREADS,
            )
    else:
        downloaded = yf.download(
            tickers=list(yahoo_to_original.keys()),
            start=start_date,
            end=(pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=False,
            actions=False,
            progress=False,
            group_by="ticker",
            threads=config.YFINANCE_THREADS,
        )

    if downloaded.empty:
        return {symbol: pd.DataFrame(columns=PRICE_COLUMNS) for symbol in symbols}

    results: dict[str, pd.DataFrame] = {}

    if isinstance(downloaded.columns, pd.MultiIndex):
        level0 = downloaded.columns.get_level_values(0)
        level1 = downloaded.columns.get_level_values(1)

        if any(value in yahoo_to_original for value in level0):
            ticker_level_first = True
        elif any(value in yahoo_to_original for value in level1):
            ticker_level_first = False
        else:
            return {symbol: pd.DataFrame(columns=PRICE_COLUMNS) for symbol in symbols}

        for yahoo_symbol, original_symbol in yahoo_to_original.items():
            try:
                if ticker_level_first:
                    frame = downloaded[yahoo_symbol].copy()
                else:
                    frame = downloaded.xs(yahoo_symbol, axis=1, level=1).copy()
            except KeyError:
                results[original_symbol] = pd.DataFrame(columns=PRICE_COLUMNS)
                continue

            if frame.empty:
                results[original_symbol] = pd.DataFrame(columns=PRICE_COLUMNS)
                continue

            frame.columns = [str(column).lower() for column in frame.columns]
            frame = frame.reset_index().rename(columns=str.lower)
            frame = frame.rename(columns={"adj close": "adj_close", "adjclose": "adj_close"})
            frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
            frame["ticker"] = original_symbol
            required = {"date", "open", "high", "low", "close", "adj_close", "volume"}
            if required.issubset(frame.columns):
                frame = frame[PRICE_COLUMNS].dropna(subset=["date", "open", "high", "low", "close"]).reset_index(drop=True)
                results[original_symbol] = frame
            else:
                results[original_symbol] = pd.DataFrame(columns=PRICE_COLUMNS)
    else:
        single_symbol = symbols[0]
        frame = downloaded.reset_index().rename(columns=str.lower)
        frame = frame.rename(columns={"adj close": "adj_close"})
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        frame["ticker"] = single_symbol
        results[single_symbol] = frame[PRICE_COLUMNS].dropna(subset=["date", "open", "high", "low", "close"]).reset_index(drop=True)

    for symbol in symbols:
        results.setdefault(symbol, pd.DataFrame(columns=PRICE_COLUMNS))

    return results


def fetch_single_ticker_history(symbol: str, start_date: str, end_date: str) -> tuple[pd.DataFrame, str]:
    """
    Fetch daily data for one ticker.

    IBKR is preferred for execution-relevant OHLCV. Yahoo is used as a fallback
    and also as a supplement for adjusted close because IBKR daily bars do not
    provide adjusted close directly.
    """
    mode = config.DATA_SOURCE_MODE.strip().lower()
    ibkr_data = pd.DataFrame()
    yahoo_data = pd.DataFrame()

    if mode == "yfinance_only":
        try:
            yahoo_data = fetch_history_yfinance(symbol, start_date, end_date)
        except Exception:
            yahoo_data = pd.DataFrame()
        if not yahoo_data.empty:
            return yahoo_data[PRICE_COLUMNS], "yfinance_only"
        return pd.DataFrame(columns=PRICE_COLUMNS), "no_data"

    try:
        ibkr_data = fetch_history_ibkr(symbol, start_date, end_date)
    except Exception:
        ibkr_data = pd.DataFrame()

    try:
        yahoo_data = fetch_history_yfinance(symbol, start_date, end_date)
    except Exception:
        yahoo_data = pd.DataFrame()

    if not ibkr_data.empty and not yahoo_data.empty:
        merged = ibkr_data.merge(
            yahoo_data[["date", "ticker", "adj_close"]],
            on=["date", "ticker"],
            how="left",
        )
        merged["adj_close"] = merged["adj_close"].where(merged["adj_close"].notna(), merged["close"])
        return merged[PRICE_COLUMNS], "ibkr+yfinance_adj_close"

    if not ibkr_data.empty:
        ibkr_data["adj_close"] = ibkr_data["close"]
        return ibkr_data[PRICE_COLUMNS], "ibkr_only"

    if not yahoo_data.empty:
        return yahoo_data[PRICE_COLUMNS], "yfinance"

    return pd.DataFrame(columns=PRICE_COLUMNS), "no_data"


def find_missing_or_incomplete_tickers(
    existing_prices: pd.DataFrame,
    required_ranges: pd.DataFrame,
) -> list[str]:
    """Return tickers missing entirely or missing the requested date coverage."""
    if required_ranges.empty:
        return []
    if existing_prices.empty:
        return required_ranges["ticker"].tolist()

    coverage = (
        existing_prices.groupby("ticker", as_index=False)
        .agg(existing_start=("date", "min"), existing_end=("date", "max"))
    )
    merged = required_ranges.merge(coverage, on="ticker", how="left")
    missing = merged[
        merged["existing_start"].isna()
        | (merged["existing_start"] > merged["required_start"])
        | (merged["existing_end"] < merged["required_end"])
    ]
    return missing["ticker"].tolist()


def load_fetch_status(path: Path = config.FETCH_STATUS_FILE) -> pd.DataFrame:
    """Load cached fetch status if it exists."""
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "ticker",
                "mode",
                "requested_start",
                "requested_end",
                "status",
                "source",
                "rows",
                "coverage_start",
                "coverage_end",
                "updated_at",
            ]
        )
    status = pd.read_csv(path)
    date_columns = ["requested_start", "requested_end", "coverage_start", "coverage_end", "updated_at"]
    for column in date_columns:
        if column in status.columns:
            status[column] = pd.to_datetime(status[column], errors="coerce")
    return status


def save_fetch_status(status: pd.DataFrame, path: Path = config.FETCH_STATUS_FILE) -> None:
    """Persist fetch status cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    status.to_csv(path, index=False)


def update_fetch_status(
    status_cache: pd.DataFrame,
    ticker: str,
    mode: str,
    requested_start: str,
    requested_end: str,
    source: str,
    rows: int,
    history: pd.DataFrame,
) -> pd.DataFrame:
    """Upsert one ticker's latest fetch result into the status cache."""
    coverage_start = pd.to_datetime(history["date"].min()) if not history.empty else pd.NaT
    coverage_end = pd.to_datetime(history["date"].max()) if not history.empty else pd.NaT
    status_value = "success" if rows > 0 else "failed"

    new_row = pd.DataFrame(
        [
            {
                "ticker": ticker,
                "mode": mode,
                "requested_start": pd.Timestamp(requested_start),
                "requested_end": pd.Timestamp(requested_end),
                "status": status_value,
                "source": source,
                "rows": rows,
                "coverage_start": coverage_start,
                "coverage_end": coverage_end,
                "updated_at": pd.Timestamp.utcnow(),
            }
        ]
    )

    if status_cache.empty:
        return new_row

    remaining = status_cache[~((status_cache["ticker"] == ticker) & (status_cache["mode"] == mode))].copy()
    return pd.concat([remaining, new_row], ignore_index=True)


def find_tickers_to_fetch(
    existing_prices: pd.DataFrame,
    required_ranges: pd.DataFrame,
    status_cache: pd.DataFrame,
    mode: str,
) -> list[str]:
    """
    Decide which tickers truly need another fetch attempt.

    Logic:
    - fetch if the ticker has no cached data at all, even if a prior attempt
      failed, because a partial/cache-corruption run may have dropped usable
      tickers from prices.csv
    - fetch if the requested end date extends beyond existing cached coverage
    - do not refetch a ticker that already failed in the same mode for the same
      requested range, because those repeated attempts are what caused the
      "downloads every run" behavior
    - do not refetch a ticker that already succeeded in the same mode unless the
      requested end date goes beyond what we already have cached
    """
    if required_ranges.empty:
        return []

    coverage = (
        existing_prices.groupby("ticker", as_index=False)
        .agg(existing_start=("date", "min"), existing_end=("date", "max"))
        if not existing_prices.empty
        else pd.DataFrame(columns=["ticker", "existing_start", "existing_end"])
    )
    merged = required_ranges.merge(coverage, on="ticker", how="left")

    latest_status = status_cache[status_cache["mode"] == mode].copy()
    if not latest_status.empty:
        latest_status = latest_status.sort_values("updated_at").drop_duplicates(subset=["ticker"], keep="last")
    merged = merged.merge(
        latest_status[["ticker", "requested_start", "requested_end", "status", "coverage_end"]],
        on="ticker",
        how="left",
        suffixes=("", "_status"),
    )

    tickers_to_fetch: list[str] = []
    for row in merged.itertuples(index=False):
        if pd.isna(row.existing_end):
            tickers_to_fetch.append(row.ticker)
            continue

        if row.existing_end < row.required_end:
            tickers_to_fetch.append(row.ticker)
            continue

        if row.existing_start > row.required_start:
            if row.status == "failed" and pd.notna(row.requested_start) and pd.notna(row.requested_end):
                if row.requested_start <= row.required_start and row.requested_end >= row.required_end:
                    continue
            tickers_to_fetch.append(row.ticker)

    return tickers_to_fetch


def fetch_prices_for_universe(
    membership: pd.DataFrame,
    start_date: str = config.START_DATE,
    end_date: str = config.END_DATE,
    output_path: Path = config.PRICES_FILE,
    existing_prices: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch prices for missing or incomplete tickers in the requested date range."""
    mode = config.DATA_SOURCE_MODE.strip().lower()
    required_ranges = expected_ticker_ranges(membership, start_date, end_date)
    existing_prices = clean_price_data(existing_prices) if existing_prices is not None and not existing_prices.empty else pd.DataFrame(columns=PRICE_COLUMNS)
    status_cache = load_fetch_status()
    tickers = find_tickers_to_fetch(existing_prices, required_ranges, status_cache, mode)
    histories: list[pd.DataFrame] = []
    stats: list[FetchStats] = []

    if mode == "yfinance_only":
        for batch_start in range(0, len(tickers), config.YFINANCE_BATCH_SIZE):
            batch = tickers[batch_start : batch_start + config.YFINANCE_BATCH_SIZE]
            batch_results = fetch_histories_yfinance_batch(batch, start_date, end_date)
            batch_histories: list[pd.DataFrame] = []
            for ticker in batch:
                history = batch_results.get(ticker, pd.DataFrame(columns=PRICE_COLUMNS))
                source = "yfinance_only"
                stats.append(
                    FetchStats(
                        ticker=ticker,
                        source=source,
                        rows=len(history),
                        status="success" if not history.empty else "failed",
                    )
                )
                status_cache = update_fetch_status(
                    status_cache=status_cache,
                    ticker=ticker,
                    mode=mode,
                    requested_start=start_date,
                    requested_end=end_date,
                    source=source,
                    rows=len(history),
                    history=history,
                )
                if not history.empty:
                    histories.append(history)
                    batch_histories.append(history)
            if batch_histories:
                batch_prices = pd.concat(batch_histories, ignore_index=True)
                existing_prices = merge_price_frames(existing_prices, batch_prices)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                existing_prices.to_csv(output_path, index=False)
            save_fetch_status(status_cache)
    else:
        for ticker in tickers:
            history, source = fetch_single_ticker_history(ticker, start_date, end_date)
            stats.append(
                FetchStats(
                    ticker=ticker,
                    source=source,
                    rows=len(history),
                    status="success" if not history.empty else "failed",
                )
            )
            status_cache = update_fetch_status(
                status_cache=status_cache,
                ticker=ticker,
                mode=mode,
                requested_start=start_date,
                requested_end=end_date,
                source=source,
                rows=len(history),
                history=history,
            )
            if not history.empty:
                histories.append(history)
                existing_prices = merge_price_frames(existing_prices, history)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                existing_prices.to_csv(output_path, index=False)
            save_fetch_status(status_cache)

    new_prices = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame(columns=PRICE_COLUMNS)
    new_prices = clean_price_data(new_prices) if not new_prices.empty else new_prices

    if not existing_prices.empty:
        # Keep all cached rows. Successful fetches have already been merged into
        # existing_prices during the loop above; failed fetches must not cause us
        # to drop the ticker's previously cached history.
        prices = existing_prices.copy()
    else:
        prices = new_prices.copy()

    prices = clean_price_data(prices) if not prices.empty else pd.DataFrame(columns=PRICE_COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(output_path, index=False)
    save_fetch_status(status_cache)
    stats_frame = pd.DataFrame(stats)
    return prices, stats_frame, status_cache


def clean_price_data(prices: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean raw price data."""
    missing = set(PRICE_COLUMNS).difference(prices.columns)
    if missing:
        raise ValueError(f"Price data is missing required columns: {sorted(missing)}")

    cleaned = prices.copy()
    cleaned["date"] = pd.to_datetime(cleaned["date"]).dt.normalize()
    cleaned["ticker"] = cleaned["ticker"].astype(str).str.strip()
    cleaned = cleaned.dropna(subset=["date", "ticker", "open", "high", "low", "close"])
    cleaned = cleaned.drop_duplicates(subset=["date", "ticker"], keep="last")
    cleaned = cleaned.sort_values(["ticker", "date"]).reset_index(drop=True)
    return cleaned


def merge_price_frames(existing_prices: pd.DataFrame, new_prices: pd.DataFrame) -> pd.DataFrame:
    """Merge newly fetched rows into the cached price table."""
    if existing_prices.empty:
        return clean_price_data(new_prices) if not new_prices.empty else pd.DataFrame(columns=PRICE_COLUMNS)
    if new_prices.empty:
        return clean_price_data(existing_prices)
    combined = pd.concat([existing_prices, new_prices], ignore_index=True)
    return clean_price_data(combined)


def load_price_data(path: Path = config.PRICES_FILE) -> pd.DataFrame:
    """Load cached price data from CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Price file not found: {path}")
    prices = pd.read_csv(path)
    return clean_price_data(prices)


def filter_prices_to_membership(prices: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """
    Filter price observations to point-in-time S&P 500 membership.

    Assumption:
    `end_date` from the interval file is treated as the first date the ticker is
    no longer eligible. That matches the usual convention that index changes
    become effective before the market opens on the stated date.
    """
    merged = prices.merge(membership, on="ticker", how="left")
    eligible = merged[
        (merged["date"] >= merged["start_date"])
        & (merged["end_date"].isna() | (merged["date"] < merged["end_date"]))
    ].copy()
    eligible = eligible.drop(columns=["start_date", "end_date"])
    eligible = eligible.drop_duplicates(subset=["date", "ticker"], keep="first")
    return eligible.sort_values(["ticker", "date"]).reset_index(drop=True)


def save_current_ticker_snapshot(
    membership: pd.DataFrame,
    as_of_date: str,
    output_path: Path,
) -> None:
    """Optional helper to save a point-in-time constituent list."""
    as_of_ts = pd.Timestamp(as_of_date)
    current = membership[
        (membership["start_date"] <= as_of_ts)
        & (membership["end_date"].isna() | (membership["end_date"] > as_of_ts))
    ]["ticker"].drop_duplicates()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    current.to_frame(name="ticker").sort_values("ticker").to_csv(output_path, index=False)


def summarize_fetch_stats(stats: pd.DataFrame) -> pd.DataFrame:
    """Aggregate data-fetch results by source."""
    if stats.empty:
        return pd.DataFrame(columns=["source", "status", "ticker_count", "total_rows"])
    return (
        stats.groupby(["source", "status"], as_index=False)
        .agg(ticker_count=("ticker", "count"), total_rows=("rows", "sum"))
        .sort_values(["status", "source"])
        .reset_index(drop=True)
    )
