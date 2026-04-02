import math
from dataclasses import dataclass, field

import pandas as pd

from .data_fetcher import add_bollinger_bands
from .strategy import TradeRecord, check_entry, check_sma_exit, check_stop_loss


@dataclass
class BacktestResult:
    trades: list[TradeRecord] = field(default_factory=list)
    total_return: float = 0.0
    win_rate: float = 0.0
    trade_count: int = 0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0


def run_backtest(
    df: pd.DataFrame,
    ma_period: int,
    num_std: float,
    stop_pct: float,
    commission: float,
) -> BacktestResult:
    """執行布林通道回測。

    信號在當日 Close 產生，次日 Open 成交。
    """
    df = add_bollinger_bands(df, ma_period, num_std)

    trades: list[TradeRecord] = []
    holding = False
    pending_entry = False
    pending_exit = False
    signal_date = ""
    buy_price = 0.0
    buy_date = ""
    exit_type = ""

    for i in range(len(df)):
        row = df.iloc[i]
        date = df.index[i]
        date_str = str(date.date()) if hasattr(date, "date") else str(date)
        close = row["Close"]
        open_price = row["Open"]
        sma = row["SMA"]
        lower = row["Lower"]

        # 次日成交：處理前一日掛單
        if pending_entry:
            pending_entry = False
            holding = True
            buy_price = open_price
            buy_date = date_str
            # 成交後同日仍需檢查出場信號
            # （但不在同日出場，出場信號最早在持倉後第一個收盤判斷）

        if pending_exit:
            pending_exit = False
            _close_trade(trades, signal_date, buy_date, buy_price,
                         date_str, open_price, exit_type, commission)
            holding = False

        if math.isnan(sma):
            continue

        if not holding:
            if check_entry(close, lower):
                pending_entry = True
                signal_date = date_str
        else:
            # 優先檢查停損
            if check_stop_loss(close, buy_price, stop_pct):
                pending_exit = True
                exit_type = "stop_loss"
                signal_date = date_str
            elif check_sma_exit(close, sma):
                pending_exit = True
                exit_type = "sma_cross"
                signal_date = date_str

    # 回測結束，強制平倉
    if holding and not pending_exit:
        last_date = str(df.index[-1].date()) if hasattr(df.index[-1], "date") else str(df.index[-1])
        last_close = df["Close"].iloc[-1]
        _close_trade(trades, last_date, buy_date, buy_price,
                     last_date, last_close, "end_of_data", commission)
    elif pending_exit:
        # 最後一日有出場信號但無次日可成交，以當日收盤平倉
        last_date = str(df.index[-1].date()) if hasattr(df.index[-1], "date") else str(df.index[-1])
        last_close = df["Close"].iloc[-1]
        _close_trade(trades, signal_date, buy_date, buy_price,
                     last_date, last_close, exit_type, commission)

    return _compute_result(trades)


def _close_trade(
    trades: list[TradeRecord],
    signal_date: str,
    buy_date: str,
    buy_price: float,
    sell_date: str,
    sell_price: float,
    exit_type: str,
    commission: float,
) -> None:
    pnl = (sell_price - buy_price) / buy_price - 2 * commission
    trades.append(
        TradeRecord(
            signal_date=signal_date,
            buy_date=buy_date,
            buy_price=buy_price,
            sell_date=sell_date,
            sell_price=sell_price,
            exit_type=exit_type,
            pnl=pnl,
        )
    )


def _compute_result(trades: list[TradeRecord]) -> BacktestResult:
    if not trades:
        return BacktestResult()

    # 累積報酬
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    gross_profit = 0.0
    gross_loss = 0.0
    wins = 0

    for t in trades:
        equity *= 1 + t.pnl
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak
        if dd > max_dd:
            max_dd = dd

        if t.pnl > 0:
            wins += 1
            gross_profit += t.pnl
        else:
            gross_loss += abs(t.pnl)

    n = len(trades)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return BacktestResult(
        trades=trades,
        total_return=equity - 1,
        win_rate=wins / n,
        trade_count=n,
        max_drawdown=max_dd,
        profit_factor=profit_factor,
    )
