from __future__ import annotations

import numpy as np
import pandas as pd


def compute_factors(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute raw factors and forward returns without look-ahead bias."""
    data = prices.copy().sort_values(["ticker", "date"]).reset_index(drop=True)
    grouped = data.groupby("ticker", group_keys=False)

    data["adj_return_1d"] = grouped["adj_close"].pct_change()
    data["ret_5d"] = grouped["adj_close"].pct_change(5)
    data["reversal_5d"] = -data["ret_5d"]

    adj_close_shift_5 = grouped["adj_close"].shift(5)
    adj_close_shift_60 = grouped["adj_close"].shift(60)
    data["momentum_60_5"] = adj_close_shift_5 / adj_close_shift_60 - 1.0

    adj_close_shift_21 = grouped["adj_close"].shift(21)
    adj_close_shift_252 = grouped["adj_close"].shift(252)
    data["momentum_252_21"] = adj_close_shift_21 / adj_close_shift_252 - 1.0

    rolling_high_252 = grouped["adj_close"].transform(lambda series: series.rolling(252).max())
    data["proximity_52w_high"] = data["adj_close"] / rolling_high_252 - 1.0

    return_126 = grouped["adj_close"].pct_change(126)
    volatility_126 = grouped["adj_return_1d"].transform(lambda series: series.rolling(126).std())
    data["trend_quality_126"] = return_126 / volatility_126.replace(0.0, np.nan)

    data["vol_20d"] = grouped["adj_return_1d"].transform(lambda series: series.rolling(20).std())
    data["low_vol_20d"] = -data["vol_20d"]

    volume_ma20 = grouped["volume"].transform(lambda series: series.rolling(20).mean())
    safe_volume = data["volume"].where(data["volume"] > 0)
    safe_volume_ma20 = volume_ma20.where(volume_ma20 > 0)
    data["volume_shock"] = np.log(safe_volume / safe_volume_ma20)

    prev_close = grouped["close"].shift(1)
    data["overnight_ret"] = data["open"] / prev_close - 1.0
    data["overnight_reversal"] = -data["overnight_ret"]

    data["entry_open"] = grouped["open"].shift(-1)
    data["hold_day_high"] = grouped["high"].shift(-1)
    data["hold_day_low"] = grouped["low"].shift(-1)
    next_open = grouped["open"].shift(-1)
    next_next_open = grouped["open"].shift(-2)
    data["forward_return"] = next_next_open / next_open - 1.0

    return data
