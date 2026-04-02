import math
from dataclasses import dataclass, field

import pandas as pd

from .data_fetcher import add_moving_average
from .strategy import (
    TradeRecord,
    check_entry,
    check_ma_cross_exit,
    check_protection_exit,
)


@dataclass
class BacktestResult:
    trades: list[TradeRecord] = field(default_factory=list)
    cumulative_return: float = 0.0
    trade_count: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    exit_type_counts: dict[str, int] = field(default_factory=dict)


def run_backtest(
    df: pd.DataFrame,
    ma_period: int,
    entry_distance: float,
    protection_pct: float,
) -> BacktestResult:
    """執行回測，回傳完整的交易紀錄與統計。"""
    df = add_moving_average(df, ma_period)

    trades: list[TradeRecord] = []
    holding = False
    buy_price = 0.0
    buy_date = ""
    protection_activated = False

    for date, row in df.iterrows():
        close = row["Close"]
        ma = row["MA"]

        if math.isnan(ma):
            continue

        if not holding:
            if check_entry(close, ma, entry_distance):
                holding = True
                buy_price = close
                buy_date = str(date.date()) if hasattr(date, "date") else str(date)
                protection_activated = False
        else:
            # 1. 更新保護狀態 & 檢查保護出場
            should_exit, protection_activated = check_protection_exit(
                close, buy_price, protection_pct, protection_activated
            )
            if should_exit:
                _close_trade(trades, buy_date, buy_price, date, close, "protection")
                holding = False
                continue

            # 2. 檢查均線交叉出場
            if check_ma_cross_exit(close, ma):
                _close_trade(trades, buy_date, buy_price, date, close, "ma_cross")
                holding = False
                continue

    # 回測結束，強制平倉
    if holding:
        last_date = df.index[-1]
        last_close = df["Close"].iloc[-1]
        _close_trade(trades, buy_date, buy_price, last_date, last_close, "end_of_data")

    return _compute_result(trades)


def _close_trade(
    trades: list[TradeRecord],
    buy_date: str,
    buy_price: float,
    sell_date,
    sell_price: float,
    exit_type: str,
) -> None:
    sell_date_str = (
        str(sell_date.date()) if hasattr(sell_date, "date") else str(sell_date)
    )
    ret = (sell_price - buy_price) / buy_price
    trades.append(
        TradeRecord(
            buy_date=buy_date,
            buy_price=buy_price,
            sell_date=sell_date_str,
            sell_price=sell_price,
            exit_type=exit_type,
            return_pct=ret,
        )
    )


def _compute_result(trades: list[TradeRecord]) -> BacktestResult:
    if not trades:
        return BacktestResult()

    cumulative = 1.0
    wins = []
    losses = []
    exit_counts: dict[str, int] = {}

    for t in trades:
        cumulative *= 1 + t.return_pct
        if t.return_pct > 0:
            wins.append(t.return_pct)
        else:
            losses.append(t.return_pct)
        exit_counts[t.exit_type] = exit_counts.get(t.exit_type, 0) + 1

    n = len(trades)
    return BacktestResult(
        trades=trades,
        cumulative_return=cumulative - 1,
        trade_count=n,
        win_rate=len(wins) / n if n else 0,
        avg_win=sum(wins) / len(wins) if wins else 0,
        avg_loss=sum(losses) / len(losses) if losses else 0,
        exit_type_counts=exit_counts,
    )
