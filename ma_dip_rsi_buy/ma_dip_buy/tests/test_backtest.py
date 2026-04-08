import pandas as pd
import pytest

from ma_dip_buy.backtest import (
    compute_max_drawdown,
    compute_trade_return,
    run_backtest,
)


class TestTradeReturn:
    def test_profitable_trade(self):
        """獲利交易：扣除交易成本後仍為正。"""
        # buy 100, sell 110
        ret = compute_trade_return(100, 110)
        # cost: buy 100.1425, sell 110*(1-0.004425) = 109.51325
        # return: (109.51325 - 100.1425) / 100.1425 ≈ 0.0936
        assert ret > 0
        assert ret < 0.10  # raw 10% minus costs

    def test_losing_trade(self):
        """虧損交易。"""
        ret = compute_trade_return(100, 90)
        assert ret < 0

    def test_breakeven_still_loses(self):
        """買賣同價因手續費仍為負。"""
        ret = compute_trade_return(100, 100)
        assert ret < 0


class TestMaxDrawdown:
    def test_no_trades(self):
        assert compute_max_drawdown([]) == 0.0

    def test_all_wins(self):
        dd = compute_max_drawdown([0.05, 0.03, 0.02])
        assert dd == 0.0

    def test_single_loss(self):
        dd = compute_max_drawdown([0.10, -0.20, 0.05])
        assert dd > 0
        assert dd < 1.0

    def test_known_drawdown(self):
        # equity: 1.0 → 1.1 → 0.88 → 0.924
        # peak = 1.1, trough = 0.88, dd = (1.1-0.88)/1.1 = 0.2
        dd = compute_max_drawdown([0.10, -0.20, 0.05])
        assert abs(dd - 0.2) < 0.001


class TestRunBacktest:
    def _make_df(self, closes):
        return pd.DataFrame({
            "Open": closes,
            "High": closes,
            "Low": closes,
            "Close": closes,
            "Volume": [1000] * len(closes),
        }, index=pd.date_range("2020-01-01", periods=len(closes)))

    def test_no_trades(self):
        """股價一直高於均線，沒有交易。"""
        closes = [110 + i for i in range(20)]
        df = self._make_df(closes)
        result = run_backtest(df, x=5, m=10, n=10, k=15, t=10)
        assert result.n_trades == 0
        assert result.score == 0.0

    def test_has_trades(self):
        """有交易產生時各欄位應合理。"""
        # 製造一個 dip-and-recover 的序列
        up = list(range(100, 110))          # warmup for SMA
        dip = [90, 88, 85, 95, 105, 102]   # dip then recover
        closes = up + dip * 3              # repeat for more trades
        df = self._make_df(closes)
        result = run_backtest(df, x=5, m=5, n=10, k=15, t=10)
        assert result.n_trades > 0
        assert 0 <= result.win_rate <= 1
        assert len(result.returns) == result.n_trades
