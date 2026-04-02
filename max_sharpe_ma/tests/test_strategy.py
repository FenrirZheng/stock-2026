import pandas as pd
import numpy as np

from ..strategy import StrategyResult, compute_positions, compute_strategy_returns


class TestComputePositions:
    def test_above_sma_gives_position_one(self):
        close = pd.Series([105.0])
        sma = pd.Series([100.0])
        pos = compute_positions(close, sma)
        assert pos.iloc[0] == 1.0

    def test_below_sma_gives_position_zero(self):
        close = pd.Series([95.0])
        sma = pd.Series([100.0])
        pos = compute_positions(close, sma)
        assert pos.iloc[0] == 0.0

    def test_equal_sma_gives_position_zero(self):
        """close == SMA → not strictly greater → position 0."""
        close = pd.Series([100.0])
        sma = pd.Series([100.0])
        pos = compute_positions(close, sma)
        assert pos.iloc[0] == 0.0

    def test_nan_sma_gives_position_zero(self):
        close = pd.Series([105.0])
        sma = pd.Series([np.nan])
        pos = compute_positions(close, sma)
        assert pos.iloc[0] == 0.0

    def test_series_mixed(self):
        close = pd.Series([90.0, 110.0, 100.0, 105.0])
        sma = pd.Series([np.nan, 100.0, 100.0, 100.0])
        pos = compute_positions(close, sma)
        expected = pd.Series([0.0, 1.0, 0.0, 1.0])
        pd.testing.assert_series_equal(pos, expected)


class TestComputeStrategyReturns:
    def test_shift_avoids_lookahead(self):
        """position[t-1] determines return[t], not position[t]."""
        #       day0    day1    day2    day3
        close = pd.Series([100.0, 110.0, 105.0, 115.0])
        # stock returns: NaN, +10%, -4.55%, +9.52%
        # positions:      0,    1,     1,     0
        positions = pd.Series([0.0, 1.0, 1.0, 0.0])
        result = compute_strategy_returns(close, positions)

        # strategy_return[1] = position[0]*stock_ret[1] = 0 * 0.10 = 0
        # strategy_return[2] = position[1]*stock_ret[2] = 1 * (-0.0455) = -0.0455
        # strategy_return[3] = position[2]*stock_ret[3] = 1 * 0.0952 = 0.0952
        assert len(result.daily_returns) == 3
        assert abs(result.daily_returns.iloc[0] - 0.0) < 1e-9
        assert abs(result.daily_returns.iloc[1] - (105 - 110) / 110) < 1e-9
        assert abs(result.daily_returns.iloc[2] - (115 - 105) / 105) < 1e-9

    def test_cash_days_return_zero(self):
        """When position is 0, strategy return is 0 regardless of stock move."""
        close = pd.Series([100.0, 120.0, 130.0])
        positions = pd.Series([0.0, 0.0, 0.0])
        result = compute_strategy_returns(close, positions)

        for r in result.daily_returns:
            assert abs(r) < 1e-9

    def test_always_holding(self):
        """When always in position, strategy return equals stock return."""
        close = pd.Series([100.0, 110.0, 121.0])
        positions = pd.Series([1.0, 1.0, 1.0])
        result = compute_strategy_returns(close, positions)
        stock_rets = close.pct_change().dropna()

        # strategy_return[1] = pos[0]*stock[1] = 1*0.10 = 0.10
        # strategy_return[2] = pos[1]*stock[2] = 1*0.10 = 0.10
        assert len(result.daily_returns) == 2
        assert abs(result.daily_returns.iloc[0] - stock_rets.iloc[0]) < 1e-9
        assert abs(result.daily_returns.iloc[1] - stock_rets.iloc[1]) < 1e-9

    def test_result_type(self):
        close = pd.Series([100.0, 110.0])
        positions = pd.Series([1.0, 1.0])
        result = compute_strategy_returns(close, positions)
        assert isinstance(result, StrategyResult)
        assert isinstance(result.daily_returns, pd.Series)
        assert isinstance(result.positions, pd.Series)
        assert isinstance(result.stock_returns, pd.Series)
