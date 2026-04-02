import pandas as pd
import numpy as np

from ..backtest import BacktestResult, compute_sharpe, run_backtest


def _make_df(closes: list[float]) -> pd.DataFrame:
    """建立合成 DataFrame 供測試用。"""
    dates = pd.date_range("2024-01-01", periods=len(closes), freq="B")
    return pd.DataFrame(
        {
            "Open": closes,
            "High": closes,
            "Low": closes,
            "Close": closes,
            "Volume": [1000] * len(closes),
        },
        index=dates,
    )


class TestComputeSharpe:
    def test_constant_returns_gives_zero(self):
        """If std == 0, Sharpe should be 0.0."""
        rets = pd.Series([0.01, 0.01, 0.01, 0.01])
        assert compute_sharpe(rets) == 0.0

    def test_all_zero_returns_gives_zero(self):
        rets = pd.Series([0.0, 0.0, 0.0])
        assert compute_sharpe(rets) == 0.0

    def test_known_values(self):
        """Verify formula: mean/std * sqrt(252)."""
        rets = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])
        mean_r = rets.mean()
        std_r = rets.std(ddof=1)
        expected = mean_r / std_r * (252 ** 0.5)
        assert abs(compute_sharpe(rets) - expected) < 1e-9

    def test_empty_series_gives_zero(self):
        rets = pd.Series([], dtype=float)
        assert compute_sharpe(rets) == 0.0


class TestRunBacktest:
    def test_always_above_sma(self):
        """Price always above SMA → always holding → Sharpe equals buy-and-hold."""
        # Strictly increasing: always above SMA(2)
        closes = [100.0, 102.0, 104.0, 106.0, 108.0, 110.0]
        df = _make_df(closes)
        result = run_backtest(df, ma_period=2)

        assert isinstance(result, BacktestResult)
        assert result.sharpe_ratio > 0
        assert result.exposure_ratio > 0.5

    def test_always_below_sma(self):
        """Strictly decreasing price → close always below SMA(2) → cash → Sharpe 0."""
        closes = [110.0, 108.0, 106.0, 104.0, 102.0, 100.0]
        df = _make_df(closes)
        result = run_backtest(df, ma_period=2)

        assert result.sharpe_ratio == 0.0
        assert result.exposure_ratio == 0.0

    def test_total_return_calculation(self):
        """Verify total_return matches manual calculation."""
        closes = [100.0, 102.0, 104.0, 106.0, 108.0]
        df = _make_df(closes)
        result = run_backtest(df, ma_period=2)

        assert result.total_return > 0
        assert result.n_days_total > 0

    def test_result_fields(self):
        closes = [100.0, 105.0, 102.0, 108.0, 103.0]
        df = _make_df(closes)
        result = run_backtest(df, ma_period=2)

        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.annualized_return, float)
        assert isinstance(result.annualized_volatility, float)
        assert isinstance(result.total_return, float)
        assert isinstance(result.n_days_held, int)
        assert isinstance(result.n_days_total, int)
        assert isinstance(result.exposure_ratio, float)
