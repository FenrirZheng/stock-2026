from dataclasses import dataclass

import pandas as pd


@dataclass
class StrategyResult:
    daily_returns: pd.Series
    positions: pd.Series
    stock_returns: pd.Series


def compute_positions(close: pd.Series, sma: pd.Series) -> pd.Series:
    """計算每日持倉：close > SMA 時持有 (1.0)，否則空手 (0.0)。"""
    position = (close > sma).astype(float)
    position[sma.isna()] = 0.0
    return position


def compute_strategy_returns(
    close: pd.Series, positions: pd.Series
) -> StrategyResult:
    """計算策略每日報酬。position[t-1] 決定 day t 的報酬，避免前視偏差。"""
    stock_returns = close.pct_change()
    strategy_returns = positions.shift(1) * stock_returns
    strategy_returns = strategy_returns.dropna()

    return StrategyResult(
        daily_returns=strategy_returns,
        positions=positions,
        stock_returns=stock_returns,
    )
