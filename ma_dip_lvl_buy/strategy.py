from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import MA_CROSS_CONFIRM_DAYS


@dataclass
class Trade:
    entry_date: str
    entry_price: float      # 隔日 Open（修正 look-ahead bias）
    signal_date: str         # 產生訊號的日期（前一日 Close 觸發）
    exit_date: str
    exit_price: float
    days_held: int
    exit_reason: str


def run_trades(df: pd.DataFrame, ma_period: int, dip_pct: float,
               rsi_threshold: float, timeout_days: int,
               hard_stop_pct: float, trail_stop_pct: float) -> list[Trade]:
    """
    均線回跌買進策略（修正版）。

    進場（隔日 Open）：
      前一日 Close < SMA × (1 - dip_pct/100) 且 RSI < rsi_threshold
      → 隔日以 Open 價進場

    出場優先順序（當日 Close）：
      1. 固定止損  Close < entry_price × (1 - hard_stop_pct/100)
      2. Trailing Stop  Close < peak_price × (1 - trail_stop_pct/100)
      3. 未突破超時  持倉天數 >= timeout_days 且從未連續站上均線
      4. 均線跌破  曾連續站上均線，今日 Close < SMA
    """
    closes = df["Close"].values
    opens = df["Open"].values
    sma = df["SMA"].values
    rsi = df["RSI"].values
    dates = df.index

    trades: list[Trade] = []
    n = len(closes)

    in_position = False
    pending_entry = False
    signal_idx = -1
    entry_price = 0.0
    entry_idx = 0
    peak_price = 0.0
    consecutive_above_ma = 0
    ever_confirmed_cross = False

    for i in range(n):
        close = float(closes[i])
        open_price = float(opens[i])
        ma = float(sma[i])
        r = float(rsi[i])

        # --- 隔日進場執行 ---
        if pending_entry and not in_position:
            in_position = True
            entry_price = open_price
            entry_idx = i
            peak_price = open_price
            consecutive_above_ma = 0
            ever_confirmed_cross = False
            pending_entry = False

        if not in_position:
            # --- 產生進場訊號（隔日才執行） ---
            if (close < ma * (1 - dip_pct / 100)
                    and r < rsi_threshold
                    and i < n - 1):  # 確保還有隔日可進場
                pending_entry = True
                signal_idx = i
            continue

        # --- 持倉中 ---
        peak_price = max(peak_price, close)

        if close >= ma:
            consecutive_above_ma += 1
        else:
            consecutive_above_ma = 0

        if consecutive_above_ma >= MA_CROSS_CONFIRM_DAYS:
            ever_confirmed_cross = True

        days_held = i - entry_idx
        exit_reason = ""

        # 出場條件（優先順序由高到低）
        if close < entry_price * (1 - hard_stop_pct / 100):
            exit_reason = "hard_stop"
        elif close < peak_price * (1 - trail_stop_pct / 100):
            exit_reason = "trailing_stop"
        elif days_held >= timeout_days and not ever_confirmed_cross:
            exit_reason = "timeout"
        elif ever_confirmed_cross and close < ma:
            exit_reason = "ma_crossback"

        if exit_reason:
            trades.append(Trade(
                signal_date=str(dates[signal_idx].date()),
                entry_date=str(dates[entry_idx].date()),
                entry_price=entry_price,
                exit_date=str(dates[i].date()),
                exit_price=close,
                days_held=days_held,
                exit_reason=exit_reason,
            ))
            in_position = False

    # 回測結束仍持倉 → 以最後一天 Close 強制平倉
    if in_position:
        last_close = float(closes[n - 1])
        trades.append(Trade(
            signal_date=str(dates[signal_idx].date()),
            entry_date=str(dates[entry_idx].date()),
            entry_price=entry_price,
            exit_date=str(dates[n - 1].date()),
            exit_price=last_close,
            days_held=n - 1 - entry_idx,
            exit_reason="end_of_data",
        ))

    return trades
