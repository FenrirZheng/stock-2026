from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import COMMISSION_RATE, TAX_RATE
from .data_fetcher import add_indicators
from .strategy import Trade, run_trades


@dataclass
class BacktestResult:
    trades: list[Trade]
    n_trades: int
    win_rate: float
    mean_return: float
    sharpe_ratio: float         # 年化 Sharpe Ratio
    max_drawdown: float         # 日頻 equity curve 計算
    total_return: float
    returns: list[float]        # 每筆交易的報酬率（扣除手續費）
    daily_equity: list[float]   # 每日淨值曲線


def compute_trade_return(entry_price: float, exit_price: float) -> float:
    """計算單筆交易報酬率，含台股交易成本。"""
    buy_cost = entry_price * (1 + COMMISSION_RATE)
    sell_revenue = exit_price * (1 - COMMISSION_RATE - TAX_RATE)
    return (sell_revenue - buy_cost) / buy_cost


def compute_daily_equity(
    df: pd.DataFrame, trades: list[Trade]
) -> list[float]:
    """
    建立每日淨值曲線。
    持倉期間用每日 Close 計算浮動損益，空倉期間淨值不變。
    """
    dates = df.index
    date_to_idx = {d: i for i, d in enumerate(dates)}
    n_days = len(dates)
    closes = df["Close"].values

    equity_curve = [1.0] * n_days
    in_trade = [False] * n_days

    # 標記哪些天在持倉中
    for tr in trades:
        entry_dt = pd.Timestamp(tr.entry_date)
        exit_dt = pd.Timestamp(tr.exit_date)
        if entry_dt not in date_to_idx or exit_dt not in date_to_idx:
            continue
        e_start = date_to_idx[entry_dt]
        e_end = date_to_idx[exit_dt]
        for j in range(e_start, e_end + 1):
            in_trade[j] = True

    current_equity = 1.0

    for tr in trades:
        entry_dt = pd.Timestamp(tr.entry_date)
        exit_dt = pd.Timestamp(tr.exit_date)
        if entry_dt not in date_to_idx or exit_dt not in date_to_idx:
            continue
        e_start = date_to_idx[entry_dt]
        e_end = date_to_idx[exit_dt]

        for j in range(e_start, e_end + 1):
            if j == e_end:
                trade_ret = compute_trade_return(tr.entry_price, tr.exit_price)
                current_equity *= (1 + trade_ret)
            else:
                if j == e_start:
                    prev_price = tr.entry_price
                else:
                    prev_price = float(closes[j - 1])
                day_close = float(closes[j])
                daily_ret = (day_close - prev_price) / prev_price
                current_equity *= (1 + daily_ret)
            equity_curve[j] = current_equity

        # 出場後填寫空倉日
        for j in range(e_end + 1, n_days):
            if in_trade[j]:
                break
            equity_curve[j] = current_equity

    # 填補初始空倉期
    last_val = 1.0
    for j in range(n_days):
        if equity_curve[j] == 1.0 and not in_trade[j] and j > 0:
            equity_curve[j] = last_val
        last_val = equity_curve[j]

    return equity_curve


def compute_max_drawdown(equity: list[float]) -> float:
    """以日頻 equity curve 計算最大回撤。"""
    if not equity:
        return 0.0
    peak = equity[0]
    max_dd = 0.0
    for val in equity:
        if val > peak:
            peak = val
        dd = (peak - val) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def compute_sharpe_ratio(returns: list[float]) -> float:
    """計算年化 Sharpe Ratio（以每筆交易報酬估算）。"""
    if len(returns) < 2:
        return 0.0
    arr = np.array(returns)
    mean_r = arr.mean()
    std_r = arr.std(ddof=1)
    if std_r == 0:
        return 0.0
    trades_per_year = max(len(returns) / 3, 1)
    return (mean_r / std_r) * np.sqrt(trades_per_year)


def run_backtest(df: pd.DataFrame, x: int, m: float, n: int,
                 k: float, t: float,
                 rsi_threshold: float) -> BacktestResult:
    """執行完整回測：加指標 → 產生交易 → 日頻淨值 → 計算指標。"""
    df_ind = add_indicators(df, x)
    trades = run_trades(df_ind, x, m, n, k, t, rsi_threshold)

    returns = [compute_trade_return(tr.entry_price, tr.exit_price)
               for tr in trades]

    n_trades = len(trades)
    total_days = len(df_ind)

    if n_trades == 0:
        return BacktestResult(
            trades=trades, n_trades=0, win_rate=0.0,
            mean_return=0.0, sharpe_ratio=0.0, max_drawdown=0.0,
            total_return=0.0, returns=[],
            daily_equity=[1.0] * total_days,
        )

    win_rate = sum(1 for r in returns if r > 0) / n_trades
    mean_return = sum(returns) / n_trades
    sharpe = compute_sharpe_ratio(returns)

    # 日頻 equity curve + max drawdown
    daily_eq = compute_daily_equity(df_ind, trades)
    max_dd = compute_max_drawdown(daily_eq)

    total_return = 1.0
    for r in returns:
        total_return *= (1 + r)
    total_return -= 1.0

    return BacktestResult(
        trades=trades,
        n_trades=n_trades,
        win_rate=win_rate,
        mean_return=mean_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        total_return=total_return,
        returns=returns,
        daily_equity=daily_eq,
    )
