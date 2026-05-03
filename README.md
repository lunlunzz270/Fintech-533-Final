# S&P 500 Sector-Neutral Walk-Forward Multi-Factor Strategy

This project implements a sector-neutral S&P 500 multi-factor stock-selection backtest. The current default strategy is a weekly, long-only, walk-forward test that ranks stocks within sectors using IC-weighted medium-term price factors.

## Strategy Overview

- Universe: S&P 500 constituents from `data/sp500_membership.csv`.
- Sector data: current S&P 500 sector classifications from `data/sp500_sectors.csv`.
- Signal timing: factors are computed using information available by the close of day `t`.
- Execution timing: positions enter at the open of day `t+1`.
- Holding return:
  - `forward_return[t] = open[t+2] / open[t+1] - 1`
- Rebalance frequency:
  - weekly, every `WEEKLY_REBALANCE_DAYS = 5` trading days.
- Portfolio construction:
  - sector-neutral ranking
  - within each sector, buy the top decile by combined score
  - equal weight all selected stocks
  - long-only, no short book
- Stop loss:
  - if next-day low breaches `entry_open * (1 - STOP_LOSS_PCT)`, return is clipped to `-STOP_LOSS_PCT`

## Walk-Forward Design

The default strategy uses walk-forward evaluation:

- `WALK_FORWARD_START = "2022-01-01"`
- train window: previous `3` years
- test window: next `6` months
- roll forward every `6` months
- factor weights are estimated only from the training window

For each fold:

1. Compute daily Spearman IC for each factor in the trailing training window.
2. Keep only positive mean IC values.
3. Normalize positive ICs into factor weights.
4. Apply those weights to the next 6-month out-of-sample period.

This avoids choosing factor weights using future data.

## Factors

The current default factor set is:

1. `momentum_252_21`
   - 12-month momentum excluding the most recent month.
2. `proximity_52w_high`
   - adjusted close divided by rolling 252-day high, minus 1.
3. `trend_quality_126`
   - 126-day return divided by 126-day daily-return volatility.
4. `momentum_60_5`
   - 60-day momentum excluding the most recent 5 trading days.

Each factor is processed cross-sectionally within `date + sector`:

- winsorize at 1st and 99th percentiles
- z-score within sector
- combine using walk-forward IC weights

## Important Data Note

The current project depends heavily on `data/prices.csv`.

If this file has incomplete coverage, the strategy result can be very wrong. A healthy sector-neutral run should usually select around `40+` stocks per rebalance. If `output/walk_forward_folds.csv` shows only around `20-25` selected stocks, the price cache is incomplete.

The code now checks sector-universe price coverage:

```python
MIN_SECTOR_PRICE_COVERAGE = 0.90
```

If coverage is below this threshold, the program stops instead of producing misleading output.

## Yahoo / yfinance Rate Limit

The project uses `yfinance` by default:

```python
DATA_SOURCE_MODE = "yfinance_only"
```

Yahoo Finance may temporarily reject many repeated requests:

```text
YFRateLimitError: Too Many Requests. Rate limited. Try after a while.
```

If this happens:

- wait 30-60 minutes
- avoid repeatedly rerunning full downloads
- try a different network or VPN
- reduce `YFINANCE_BATCH_SIZE`
- use IBKR or another data source if available

When rate-limited, missing tickers such as `JPM`, `MSFT`, `NVDA`, or `NFLX` may fail to download. Do not trust backtest results generated from an incomplete universe.

## Project Layout

```text
project/
    data/
        benchmark_prices.csv
        fetch_status.csv
        prices.csv
        sp500_membership.csv
        sp500_sectors.csv
        sp500_tickers.csv
    src/
        backtest.py
        config.py
        data_loader.py
        factors.py
        ic_analysis.py
        main.py
        metrics.py
        plotting.py
        preprocessing.py
        quantile_analysis.py
        walk_forward.py
    output/
        blotter.csv
        cumulative_returns.png
        daily_portfolio_returns.csv
        drawdown.png
        factor_data.csv
        ic_summary.csv
        ledger.csv
        long_short_legs.png
        performance_summary.csv
        walk_forward_factor_weights.csv
        walk_forward_folds.csv
```

## Main Configuration

Edit `src/config.py` to change strategy settings.

Important fields:

```python
START_DATE = "2018-01-01"
END_DATE = "2026-05-01"

STRATEGY_MODE = "walk_forward_sector_neutral"
WALK_FORWARD_START = "2022-01-01"
WALK_FORWARD_TRAIN_YEARS = 3
WALK_FORWARD_TEST_MONTHS = 6
WEEKLY_REBALANCE_DAYS = 5

PORTFOLIO_SELECTION_FRACTION = 0.10
ENABLE_STOP_LOSS = True
STOP_LOSS_PCT = 0.02

MIN_SECTOR_PRICE_COVERAGE = 0.90
```

Default factors:

```python
RAW_FACTOR_COLUMNS = [
    "momentum_252_21",
    "proximity_52w_high",
    "trend_quality_126",
    "momentum_60_5",
]
```

## How To Run

From the project directory:

```bash
'/Users/jinxin/Desktop/Fintech 533/.venv/bin/python' src/main.py
```

Or from the parent folder:

```bash
cd '/Users/jinxin/Desktop/Fintech 533/project'
'/Users/jinxin/Desktop/Fintech 533/.venv/bin/python' src/main.py
```

## Output Files

### `output/performance_summary.csv`

Stitched out-of-sample performance for:

- strategy
- SPY benchmark
- excess return

Metrics include:

- cumulative return
- annualized return
- annualized volatility
- Sharpe ratio
- max drawdown
- hit rate

### `output/walk_forward_folds.csv`

One row per 6-month out-of-sample fold:

- train start/end
- test fold period
- number of rebalance days
- strategy cumulative return
- benchmark cumulative return
- excess cumulative return
- strategy Sharpe
- excess Sharpe
- average number of selected stocks

Use `average_number_of_longs` to check whether the universe is healthy.

### `output/walk_forward_factor_weights.csv`

Factor weights used in each fold. These weights are based only on the trailing 3-year training ICs.

### `output/daily_portfolio_returns.csv`

One row per rebalance date:

- strategy return
- benchmark return
- excess return
- number of selected stocks
- number of stop-loss hits
- average selected score
- fold label

### `output/blotter.csv`

One row per selected stock:

- signal date
- ticker
- sector
- side
- entry date
- exit date
- entry price
- exit price
- portfolio weight
- combined score
- raw forward return
- stop-loss-adjusted return
- stop-loss flag
- fold label

### `output/ledger.csv`

Daily aggregate portfolio ledger:

- portfolio value
- daily PnL
- strategy return
- benchmark return
- excess return
- number of longs
- number of stop-loss hits
- average long score

## Timing Alignment

The strategy avoids look-ahead in factor and return construction:

- factors use same-day close and lagged historical values
- signals are observed after close `t`
- entries happen at open `t+1`
- exits happen at open `t+2`

This is implemented in `src/factors.py`:

```python
next_open = open.shift(-1)
next_next_open = open.shift(-2)
forward_return = next_next_open / next_open - 1
```

## Current Limitations

- Sector classifications are current S&P 500 classifications, not historical point-in-time sector classifications.
- Transaction costs and slippage are not included.
- The strategy is a sparse weekly signal test: it trades only on rebalance dates and holds one open-to-open day.
- yfinance is not a production-grade data source and can be rate-limited.
- If `prices.csv` is incomplete, results are not reliable.

## Interpreting Results

Do not use the output if:

- many tickers failed in `output/fetch_failures.csv`
- `average_number_of_longs` is around `20-25` instead of roughly `40+`
- the terminal reports low sector price coverage
- yfinance recently returned `Too Many Requests`

In that case, wait for the Yahoo rate limit to clear, restore a fuller `prices.csv`, or switch to a more reliable data source.
