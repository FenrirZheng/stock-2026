from dataclasses import dataclass

import pandas as pd

from .config import COMMISSION_RATE, TAX_RATE
from .data_fetcher import add_sma
from .strategy import Trade, run_trades


@dataclass
class BacktestResult:
    trades: list[Trade]
    n_trades: int
    win_rate: float
    mean_return: float
    score: float            # win_rate × mean_return
    max_drawdown: float
    total_return: float
    returns: list[float]    # 每筆交易的報酬率（扣除手續費）


def compute_trade_return(entry_price: float, exit_price: float) -> float:
    """計算單筆交易報酬率，含台股交易成本。"""
    buy_cost = entry_price * (1 + COMMISSION_RATE)
    sell_revenue = exit_price * (1 - COMMISSION_RATE - TAX_RATE)
    return (sell_revenue - buy_cost) / buy_cost


def compute_max_drawdown(returns: list[float]) -> float:
    """以交易序列的累積淨值計算最大回撤。"""
    if not returns:
        return 0.0
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in returns:
        equity *= (1 + r)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak
        if dd > max_dd:
            max_dd = dd
    return max_dd


def run_backtest(df: pd.DataFrame, x: int, m: float, n: int,
                 k: float, t: float) -> BacktestResult:
    """執行完整回測：加 SMA → 產生交易 → 計算指標。"""
    df_sma = add_sma(df, x)
    trades = run_trades(df_sma, x, m, n, k, t)

    returns = [compute_trade_return(tr.entry_price, tr.exit_price)
               for tr in trades]

    n_trades = len(trades)
    if n_trades == 0:
        return BacktestResult(
            trades=trades, n_trades=0, win_rate=0.0,
            mean_return=0.0, score=0.0, max_drawdown=0.0,
            total_return=0.0, returns=[],
        )

    win_rate = sum(1 for r in returns if r > 0) / n_trades
    mean_return = sum(returns) / n_trades
    max_dd = compute_max_drawdown(returns)
    total_return = 1.0
    for r in returns:
        total_return *= (1 + r)
    total_return -= 1.0

    return BacktestResult(
        trades=trades,
        n_trades=n_trades,
        win_rate=win_rate,
        mean_return=mean_return,
        score=win_rate * mean_return,
        max_drawdown=max_dd,
        total_return=total_return,
        returns=returns,
    )
