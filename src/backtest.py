from __future__ import annotations

import math

import pandas as pd

import config


def construct_portfolio(factor_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a daily portfolio from combined scores using config-driven construction."""
    positions: list[pd.DataFrame] = []
    daily_rows: list[dict[str, float | int | str]] = []
    portfolio_mode = config.PORTFOLIO_MODE.strip().lower()
    enable_stop_loss = bool(config.ENABLE_STOP_LOSS)
    stop_loss_pct = float(config.STOP_LOSS_PCT)

    for date, daily in factor_data.groupby("date", sort=True):
        eligible = daily.dropna(subset=["combined_score", "forward_return"]).copy()
        if len(eligible) < config.MIN_CROSS_SECTION_SIZE:
            continue

        n_select = max(1, math.floor(len(eligible) * config.PORTFOLIO_SELECTION_FRACTION))
        ranked = eligible.sort_values(["combined_score", "ticker"]).reset_index(drop=True)
        longs = ranked.tail(n_select).copy()
        shorts = ranked.head(n_select).copy() if portfolio_mode == "long_short" else pd.DataFrame(columns=ranked.columns)

        if longs.empty or (portfolio_mode == "long_short" and shorts.empty):
            continue

        if portfolio_mode == "long_only" and enable_stop_loss:
            required_stop_columns = {"entry_open", "hold_day_low"}
            if required_stop_columns.issubset(longs.columns):
                stop_hit = longs["hold_day_low"] <= longs["entry_open"] * (1.0 - stop_loss_pct)
                longs.loc[stop_hit, "forward_return"] = -stop_loss_pct

        long_weight_total = 1.0
        longs["weight"] = long_weight_total / len(longs)
        longs["side"] = "long"
        if not shorts.empty:
            shorts["weight"] = -1.0 / len(shorts)
            shorts["side"] = "short"

        daily_positions = pd.concat([longs, shorts], ignore_index=True) if not shorts.empty else longs.copy()
        positions.append(
            daily_positions[
                ["date", "ticker", "weight", "side", "combined_score", "forward_return"]
            ].copy()
        )

        long_leg_return = float(longs["forward_return"].mean())
        short_leg_return = float(shorts["forward_return"].mean()) if not shorts.empty else 0.0
        long_short_spread = long_leg_return - short_leg_return if not shorts.empty else long_leg_return
        gross_return = float((daily_positions["weight"] * daily_positions["forward_return"]).sum())
        borrow_cost = config.SHORT_BORROW_FEE / config.ANNUALIZATION_DAYS if not shorts.empty else 0.0
        net_return = gross_return - borrow_cost

        daily_rows.append(
            {
                "date": date,
                "gross_return": gross_return,
                "borrow_cost": borrow_cost,
                "net_return": net_return,
                "long_leg_return": long_leg_return,
                "short_leg_return": short_leg_return,
                "long_short_spread": long_short_spread,
                "number_of_longs": int(len(longs)),
                "number_of_shorts": int(len(shorts)),
                "average_long_score": float(longs["combined_score"].mean()),
                "average_short_score": float(shorts["combined_score"].mean()) if not shorts.empty else float("nan"),
            }
        )

    positions_frame = (
        pd.concat(positions, ignore_index=True)
        if positions
        else pd.DataFrame(columns=["date", "ticker", "weight", "side", "combined_score", "forward_return"])
    )
    daily_returns = (
        pd.DataFrame(daily_rows).sort_values("date").reset_index(drop=True)
        if daily_rows
        else pd.DataFrame(
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
                "average_long_score",
                "average_short_score",
            ]
        )
    )

    return daily_returns, positions_frame
