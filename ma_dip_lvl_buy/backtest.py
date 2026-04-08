from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import COMMISSION_RATE, SLIPPAGE_RATE, TAX_RATE
from .data_fetcher import add_indicators
from .strategy import Trade, run_trades


@dataclass
class BacktestResult:
    trades: list[Trade]
    n_trades: int
    win_rate: float
    mean_return: float
    total_return: float
    max_drawdown: float         # 日頻 equity curve 計算
    sharpe_ratio: float         # 年化 Sharpe Ratio
    profit_factor: float        # 總獲利 / 總虧損
    exposure_ratio: float       # 在場天數 / 總天數
    annualized_return: float
    annualized_volatility: float
    returns: list[float]        # 每筆交易的報酬率
    daily_equity: list[float]   # 每日淨值曲線


def compute_trade_return(entry_price: float, exit_price: float) -> float:
    """計算單筆交易報酬率，含台股交易成本 + 滑價。"""
    buy_cost = entry_price * (1 + COMMISSION_RATE + SLIPPAGE_RATE)
    sell_revenue = exit_price * (1 - COMMISSION_RATE - TAX_RATE - SLIPPAGE_RATE)
    return (sell_revenue - buy_cost) / buy_cost


def compute_daily_equity(
    df: pd.DataFrame, trades: list[Trade]
) -> list[float]:
    """
    建立每日淨值曲線。
    持倉期間用每日 Close 計算浮動損益，空倉期間淨值不變。
    使用 NaN 標記未填充日，避免與合法淨值 1.0 混淆。
    """
    dates = df.index
    date_to_idx = {d: i for i, d in enumerate(dates)}
    n_days = len(dates)
    closes = df["Close"].values

    equity = np.full(n_days, np.nan)
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
                day_close = float(closes[j])
                prev_price = tr.entry_price if j == e_start else float(closes[j - 1])
                daily_ret = (day_close - prev_price) / prev_price
                current_equity *= (1 + daily_ret)
            equity[j] = current_equity

    # Forward-fill NaN（空倉日）with last known equity
    last_val = 1.0
    for j in range(n_days):
        if np.isnan(equity[j]):
            equity[j] = last_val
        else:
            last_val = equity[j]

    return equity.tolist()


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


def compute_sharpe_ratio(daily_equity: list[float]) -> float:
    """
    以日頻淨值曲線計算年化 Sharpe Ratio。
    Sharpe = mean(daily_returns) / std(daily_returns) × √252
    """
    if len(daily_equity) < 2:
        return 0.0
    eq = np.array(daily_equity)
    daily_returns = eq[1:] / eq[:-1] - 1
    std_r = daily_returns.std(ddof=1)
    if std_r == 0:
        return 0.0
    return (daily_returns.mean() / std_r) * np.sqrt(252)


def compute_profit_factor(returns: list[float]) -> float:
    """總獲利 / 總虧損，無虧損回傳 inf。"""
    gross_profit = sum(r for r in returns if r > 0)
    gross_loss = abs(sum(r for r in returns if r < 0))
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def run_backtest(df: pd.DataFrame, ma_period: int, dip_pct: float,
                 rsi_threshold: float, timeout_days: int,
                 hard_stop_pct: float, trail_stop_pct: float) -> BacktestResult:
    """執行完整回測：加指標 → 產生交易 → 日頻淨值 → 計算指標。"""
    df_ind = add_indicators(df, ma_period)
    trades = run_trades(df_ind, ma_period, dip_pct, rsi_threshold,
                        timeout_days, hard_stop_pct, trail_stop_pct)

    returns = [compute_trade_return(tr.entry_price, tr.exit_price)
               for tr in trades]

    n_trades = len(trades)
    total_days = len(df_ind)

    if n_trades == 0:
        return BacktestResult(
            trades=trades, n_trades=0, win_rate=0.0,
            mean_return=0.0, total_return=0.0, max_drawdown=0.0,
            sharpe_ratio=0.0, profit_factor=0.0, exposure_ratio=0.0,
            annualized_return=0.0, annualized_volatility=0.0,
            returns=[], daily_equity=[1.0] * total_days,
        )

    # 基本指標
    win_rate = sum(1 for r in returns if r > 0) / n_trades
    mean_return = sum(returns) / n_trades
    total_return = 1.0
    for r in returns:
        total_return *= (1 + r)
    total_return -= 1.0

    # 日頻 equity curve + max drawdown
    daily_eq = compute_daily_equity(df_ind, trades)
    max_dd = compute_max_drawdown(daily_eq)

    # Sharpe Ratio（基於日頻報酬）
    sharpe = compute_sharpe_ratio(daily_eq)

    # Profit Factor
    pf = compute_profit_factor(returns)

    # Exposure Ratio（在場天數 / 總天數）
    days_in_market = sum(tr.days_held for tr in trades)
    exposure = days_in_market / total_days if total_days > 0 else 0.0

    # 年化報酬 & 波動度（基於日頻淨值曲線）
    years = total_days / 252
    annualized_ret = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
    eq_arr = np.array(daily_eq)
    daily_rets = eq_arr[1:] / eq_arr[:-1] - 1
    annualized_vol = daily_rets.std(ddof=1) * np.sqrt(252) if len(daily_rets) > 1 else 0.0

    return BacktestResult(
        trades=trades,
        n_trades=n_trades,
        win_rate=win_rate,
        mean_return=mean_return,
        total_return=total_return,
        max_drawdown=max_dd,
        sharpe_ratio=sharpe,
        profit_factor=pf,
        exposure_ratio=exposure,
        annualized_return=annualized_ret,
        annualized_volatility=annualized_vol,
        returns=returns,
        daily_equity=daily_eq,
    )
