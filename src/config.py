from __future__ import annotations

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"

MEMBERSHIP_FILE = DATA_DIR / "sp500_membership.csv"
PRICES_FILE = DATA_DIR / "prices.csv"
BENCHMARK_FILE = DATA_DIR / "benchmark_prices.csv"
FETCH_STATUS_FILE = DATA_DIR / "fetch_status.csv"
SECTOR_FILE = DATA_DIR / "sp500_sectors.csv"
FACTOR_DATA_FILE = OUTPUT_DIR / "factor_data.csv"
DAILY_RETURNS_FILE = OUTPUT_DIR / "daily_portfolio_returns.csv"
BLOTTER_FILE = OUTPUT_DIR / "blotter.csv"
LEDGER_FILE = OUTPUT_DIR / "ledger.csv"
PERFORMANCE_SUMMARY_FILE = OUTPUT_DIR / "performance_summary.csv"
IC_SUMMARY_FILE = OUTPUT_DIR / "ic_summary.csv"
QUANTILE_SUMMARY_FILE = OUTPUT_DIR / "quantile_summary.csv"
FETCH_SUMMARY_FILE = OUTPUT_DIR / "fetch_summary.csv"
FETCH_FAILURES_FILE = OUTPUT_DIR / "fetch_failures.csv"
WALK_FORWARD_SUMMARY_FILE = OUTPUT_DIR / "walk_forward_summary.csv"
WALK_FORWARD_FOLDS_FILE = OUTPUT_DIR / "walk_forward_folds.csv"
WALK_FORWARD_WEIGHTS_FILE = OUTPUT_DIR / "walk_forward_factor_weights.csv"

CUMULATIVE_RETURNS_PLOT = OUTPUT_DIR / "cumulative_returns.png"
DRAWDOWN_PLOT = OUTPUT_DIR / "drawdown.png"
IC_TIMESERIES_PLOT = OUTPUT_DIR / "ic_timeseries_combined.png"
QUANTILE_PLOT = OUTPUT_DIR / "quantile_cumulative_returns.png"
LONG_SHORT_LEGS_PLOT = OUTPUT_DIR / "long_short_legs.png"

START_DATE = "2018-01-01"
END_DATE = "2026-05-01"
IN_SAMPLE_START = "2019-01-01"
IN_SAMPLE_END = "2024-12-31"
OUT_OF_SAMPLE_START = "2025-01-01"
OUT_OF_SAMPLE_END = "2026-05-01"

ANNUALIZATION_DAYS = 252
INITIAL_CAPITAL = 100000.0
SHORT_BORROW_FEE = 0.06
MIN_CROSS_SECTION_SIZE = 100
PORTFOLIO_MODE = "long_only"
PORTFOLIO_SELECTION_FRACTION = 0.10
ENABLE_STOP_LOSS = True
STOP_LOSS_PCT = 0.03
NUM_QUANTILES = 5
MIN_VALID_FACTORS = 2
WINSOR_LOWER = 0.01
WINSOR_UPPER = 0.99
ROLLING_IC_WINDOW = 60
STRATEGY_MODE = "walk_forward_sector_neutral"
WALK_FORWARD_START = "2022-01-01"
WALK_FORWARD_TRAIN_YEARS = 3
WALK_FORWARD_TEST_MONTHS = 6
WEEKLY_REBALANCE_DAYS = 1
NEUTRALIZATION_GROUP = "sector"
MIN_SECTOR_PRICE_COVERAGE = 0.90

MEMBERSHIP_SOURCE_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/sp500_ticker_start_end.csv"
)
SECTOR_SOURCE_URL = (
    "https://datahub.io/core/s-and-p-500-companies/_r/-/data/constituents.csv"
)

DATA_SOURCE_MODE = "yfinance_only"
AUTO_FETCH_MISSING_DATA = True
BENCHMARK_TICKER = "SPY"

IBKR_HOST = "127.0.0.1"
IBKR_PORT = 7497
IBKR_CLIENT_ID = 733
IBKR_TIMEOUT = 10
IBKR_USE_RTH = True

YFINANCE_THREADS = False
YFINANCE_BATCH_SIZE = 100
QUIET_YFINANCE_OUTPUT = True

RAW_FACTOR_COLUMNS = [
    "momentum_252_21",
    "proximity_52w_high",
    "trend_quality_126",
    "momentum_60_5",
]
Z_FACTOR_COLUMNS = [f"{column}_z" for column in RAW_FACTOR_COLUMNS]


def ensure_directories() -> None:
    """Create data and output directories if they do not already exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
