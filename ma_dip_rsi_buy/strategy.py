from dataclasses import dataclass

import pandas as pd


@dataclass
class Trade:
    entry_price: float
    entry_day: int       # DataFrame 中的 row index
    entry_date: str
    exit_price: float
    exit_day: int
    exit_date: str
    exit_reason: str


def run_trades(df: pd.DataFrame, x: int, m: float, n: int,
               k: float, t: float, rsi_threshold: float) -> list[Trade]:
    """
    均線回跌買進策略（狀態機）。

    進場：Close < SMA(x) × (1 - m/100) 且 RSI < rsi_threshold
    出場優先順序：
      1. 固定止損  Close < entry_price × (1 - k/100)
      2. Trailing Stop  Close < peak_price × (1 - t/100)
      3. 未突破超時  持倉天數 >= n 且從未 Close >= SMA
      4. 突破後跌破均線  曾 Close >= SMA 且今日 Close < SMA
    """
    closes = df["Close"].values
    sma = df["SMA"].values
    rsi = df["RSI"].values
    dates = df.index
    trades: list[Trade] = []

    in_position = False
    entry_price = 0.0
    entry_idx = 0
    peak_price = 0.0
    ever_crossed_ma = False

    for i in range(len(closes)):
        close = float(closes[i])
        ma = float(sma[i])
        rsi_val = float(rsi[i])

        if not in_position:
            # --- 等待進場 ---
            if close < ma * (1 - m / 100) and rsi_val < rsi_threshold:
                in_position = True
                entry_price = close
                entry_idx = i
                peak_price = close
                ever_crossed_ma = False
            continue

        # --- 持倉中 ---
        peak_price = max(peak_price, close)

        if close >= ma:
            ever_crossed_ma = True

        days_held = i - entry_idx
        exit_reason = ""

        # 出場條件（優先順序由高到低）
        if close < entry_price * (1 - k / 100):
            exit_reason = "hard_stop"
        elif close < peak_price * (1 - t / 100):
            exit_reason = "trailing_stop"
        elif days_held >= n and not ever_crossed_ma:
            exit_reason = "timeout"
        elif ever_crossed_ma and close < ma:
            exit_reason = "ma_crossback"

        if exit_reason:
            trades.append(Trade(
                entry_price=entry_price,
                entry_day=entry_idx,
                entry_date=str(dates[entry_idx].date()),
                exit_price=close,
                exit_day=i,
                exit_date=str(dates[i].date()),
                exit_reason=exit_reason,
            ))
            in_position = False

    return trades
