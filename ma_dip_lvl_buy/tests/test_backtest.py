import numpy as np
import pandas as pd
import pytest

from ma_dip_lvl_buy.backtest import (
    compute_daily_equity,
    compute_max_drawdown,
    compute_profit_factor,
    compute_sharpe_ratio,
    compute_trade_return,
)
from ma_dip_lvl_buy.strategy import Trade


def _make_df(closes, start="2024-01-02"):
    n = len(closes)
    dates = pd.bdate_range(start, periods=n)
    return pd.DataFrame({
        "Close": closes,
        "Open": closes,
        "High": closes,
        "Low": closes,
    }, index=dates)


class TestComputeTradeReturn:

    def test_profitable_trade(self):
        ret = compute_trade_return(100.0, 110.0)
        # buy_cost = 100 * (1+0.001425+0.001) = 100.2425
        # sell_rev = 110 * (1-0.001425-0.003-0.001) = 110 * 0.994575 = 109.40325
        # return = (109.40325 - 100.2425) / 100.2425
        expected = (109.40325 - 100.2425) / 100.2425
        assert abs(ret - expected) < 1e-8

    def test_losing_trade(self):
        ret = compute_trade_return(100.0, 90.0)
        assert ret < 0

    def test_flat_trade_loses_to_costs(self):
        ret = compute_trade_return(100.0, 100.0)
        assert ret < 0  # 手續費+稅+滑價 → 平盤也虧


class TestComputeDailyEquity:

    def test_no_trades(self):
        df = _make_df([100.0, 101.0, 102.0])
        equity = compute_daily_equity(df, [])
        assert equity == [1.0, 1.0, 1.0]

    def test_single_trade_equity_ends_correctly(self):
        """單筆交易結束後淨值應反映交易損益。"""
        closes = [100.0, 105.0, 110.0, 108.0, 108.0]
        df = _make_df(closes)
        dates = df.index

        trade = Trade(
            signal_date=str(dates[0].date()),
            entry_date=str(dates[1].date()),
            entry_price=105.0,
            exit_date=str(dates[2].date()),
            exit_price=110.0,
            days_held=1,
            exit_reason="trailing_stop",
        )
        equity = compute_daily_equity(df, [trade])

        # exit day equity should reflect trade return with costs
        trade_ret = compute_trade_return(105.0, 110.0)
        assert abs(equity[2] - (1.0 * (1 + trade_ret))) < 1e-8

        # post-trade days should hold flat
        assert equity[3] == equity[2]
        assert equity[4] == equity[2]

    def test_equity_1_0_not_misidentified(self):
        """淨值剛好回到 1.0 不應被誤判為未填充（NaN sentinel 修正）。"""
        # Create a trade that results in equity returning to ~1.0
        closes = [100.0, 100.0, 100.0, 100.0, 100.0]
        df = _make_df(closes)
        dates = df.index

        trade = Trade(
            signal_date=str(dates[0].date()),
            entry_date=str(dates[1].date()),
            entry_price=100.0,
            exit_date=str(dates[2].date()),
            exit_price=100.0,
            days_held=1,
            exit_reason="timeout",
        )
        equity = compute_daily_equity(df, [trade])

        # After a flat trade (with costs), equity should be < 1.0, not reset
        assert equity[2] < 1.0
        # Post-trade should carry forward the same value, not reset to 1.0
        assert equity[3] == equity[2]
        assert equity[4] == equity[2]


class TestComputeSharpeRatio:

    def test_flat_equity_returns_zero(self):
        equity = [1.0, 1.0, 1.0, 1.0]
        assert compute_sharpe_ratio(equity) == 0.0

    def test_steadily_growing_equity_positive_sharpe(self):
        # 每天漲 0.1%
        equity = [1.0 * (1.001 ** i) for i in range(100)]
        sharpe = compute_sharpe_ratio(equity)
        assert sharpe > 0

    def test_single_point_returns_zero(self):
        assert compute_sharpe_ratio([1.0]) == 0.0

    def test_uses_sqrt_252_annualization(self):
        """確認用 √252 年化，結果與手動計算一致。"""
        rng = np.random.RandomState(42)
        daily_rets = rng.normal(0.001, 0.01, 252)
        equity = [1.0]
        for r in daily_rets:
            equity.append(equity[-1] * (1 + r))

        sharpe = compute_sharpe_ratio(equity)
        # 手動計算驗證
        mean_r = daily_rets.mean()
        std_r = daily_rets.std(ddof=1)
        expected = (mean_r / std_r) * np.sqrt(252)
        assert abs(sharpe - expected) < 1e-6


class TestComputeMaxDrawdown:

    def test_no_drawdown(self):
        equity = [1.0, 1.1, 1.2, 1.3]
        assert compute_max_drawdown(equity) == 0.0

    def test_known_drawdown(self):
        equity = [1.0, 1.2, 0.9, 1.1]
        # Peak 1.2, trough 0.9 → dd = 0.3/1.2 = 0.25
        assert abs(compute_max_drawdown(equity) - 0.25) < 1e-8

    def test_empty(self):
        assert compute_max_drawdown([]) == 0.0


class TestComputeProfitFactor:

    def test_all_wins(self):
        assert compute_profit_factor([0.1, 0.2, 0.05]) == float("inf")

    def test_all_losses(self):
        assert compute_profit_factor([-0.1, -0.2]) == 0.0

    def test_mixed(self):
        # profit=0.3, loss=0.1 → pf=3.0
        pf = compute_profit_factor([0.1, 0.2, -0.1])
        assert abs(pf - 3.0) < 1e-8
