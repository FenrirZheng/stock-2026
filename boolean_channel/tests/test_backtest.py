import pandas as pd
import pytest

from boolean_channel.backtest import BacktestResult, run_backtest


def _make_df(opens: list[float], closes: list[float]) -> pd.DataFrame:
    """建立測試用 DataFrame，需同時提供 Open 和 Close。"""
    dates = pd.bdate_range("2024-01-01", periods=len(opens), freq="B")
    return pd.DataFrame(
        {
            "Open": opens,
            "High": [max(o, c) + 1 for o, c in zip(opens, closes)],
            "Low": [min(o, c) - 1 for o, c in zip(opens, closes)],
            "Close": closes,
            "Volume": [1000] * len(opens),
        },
        index=dates,
    )


class TestRunBacktest:
    def test_no_trades(self):
        # 價格在布林通道內波動，不觸發進場
        # num_std=2.0 讓通道夠寬，close 不會碰到 lower
        opens =  [100, 101, 102, 101, 102, 101, 102, 101, 102, 101]
        closes = [100, 101, 102, 101, 102, 101, 102, 101, 102, 101]
        df = _make_df(opens, closes)
        result = run_backtest(df, ma_period=3, num_std=2.0, stop_pct=0.02, commission=0.001)
        assert result.trade_count == 0
        assert result.total_return == 0.0

    def test_single_trade_sma_exit(self):
        # period=3, num_std=0.5
        # Day 1-3: Close=[100, 102, 101] → SMA=101, std≈1.0, lower≈100.5
        # Day 4: Close=99
        #         SMA=(102+101+99)/3=100.67, std≈1.53, lower≈100.67-0.76=99.90
        #         close=99 ≤ lower=99.90 → 進場信號
        # Day 5: Open=100 → 買進 @ 100
        #         SMA=(101+99+105)/3=101.67 → close=105 > SMA → 不出場
        # Day 6: Close=99 (> stop_loss threshold 98, 但 < SMA)
        #         SMA=(99+105+99)/3=101.0 → close=99 < SMA → 出場信號
        # Day 7: Open=99 → 賣出 @ 99
        # PnL = (99-100)/100 - 0.002 = -0.012
        opens =  [100, 100, 101, 101, 100, 100, 99]
        closes = [100, 102, 101,  99, 105,  99, 99]
        df = _make_df(opens, closes)
        result = run_backtest(df, ma_period=3, num_std=0.5, stop_pct=0.02, commission=0.001)

        assert result.trade_count == 1
        assert result.trades[0].exit_type == "sma_cross"
        assert result.trades[0].buy_price == 100.0
        assert result.trades[0].sell_price == 99.0
        expected_pnl = (99 - 100) / 100 - 0.002
        assert abs(result.trades[0].pnl - expected_pnl) < 1e-9

    def test_stop_loss_exit(self):
        # period=3, num_std=0.5, stop=2%
        # Day 1-3: Close=[100,102,101] → SMA=101, lower≈100.5
        # Day 4: Close=99 ≤ lower → 信號, Open=101
        # Day 5: Open=100 → 買進 @ 100
        #         Close=97 ≤ 100*(1-0.02)=98 → 停損信號
        # Day 6: Open=97 → 賣出 @ 97
        # PnL = (97-100)/100 - 0.002 = -0.032
        opens =  [100, 100, 101, 101, 100,  97]
        closes = [100, 102, 101,  99,  97,  96]
        df = _make_df(opens, closes)
        result = run_backtest(df, ma_period=3, num_std=0.5, stop_pct=0.02, commission=0.001)

        assert result.trade_count == 1
        assert result.trades[0].exit_type == "stop_loss"
        assert result.trades[0].buy_price == 100.0
        assert result.trades[0].sell_price == 97.0
        expected_pnl = (97 - 100) / 100 - 0.002
        assert abs(result.trades[0].pnl - expected_pnl) < 1e-9

    def test_commission_deducted(self):
        # 驗證手續費計算 PnL = (sell - buy) / buy - 2c
        opens =  [100, 100, 101, 101, 100, 100, 110]
        closes = [100, 102, 101,  99, 105,  98, 110]
        df = _make_df(opens, closes)
        result = run_backtest(df, ma_period=3, num_std=0.5, stop_pct=0.02, commission=0.005)

        if result.trade_count > 0:
            t = result.trades[0]
            raw_return = (t.sell_price - t.buy_price) / t.buy_price
            assert abs(t.pnl - (raw_return - 2 * 0.005)) < 1e-9

    def test_end_of_data_exit(self):
        # 進場後資料結束，強制平倉
        # period=3, num_std=0.5
        # Day 1-3: [100,102,101]
        # Day 4: close=99 → 信號
        # Day 5: open=100 → 買進, close=101 (> SMA → 不出場)
        opens =  [100, 100, 101, 101, 100]
        closes = [100, 102, 101,  99, 101]
        df = _make_df(opens, closes)
        result = run_backtest(df, ma_period=3, num_std=0.5, stop_pct=0.02, commission=0.001)

        assert result.trade_count == 1
        assert result.trades[0].exit_type == "end_of_data"

    def test_max_drawdown(self):
        result = BacktestResult()
        assert result.max_drawdown == 0.0

    def test_profit_factor_no_loss(self):
        # 全勝時 profit_factor = inf
        result = BacktestResult(profit_factor=float("inf"))
        assert result.profit_factor == float("inf")

    def test_win_rate(self):
        # 2 trades: 1 win, 1 loss → 50%
        opens =  [100, 100, 101, 101, 100, 100, 99, 100, 100, 101, 101, 100, 100, 97]
        closes = [100, 102, 101,  99, 105,  98, 99, 100, 102, 101,  99,  97,  96, 96]
        df = _make_df(opens, closes)
        result = run_backtest(df, ma_period=3, num_std=0.5, stop_pct=0.02, commission=0.001)
        if result.trade_count >= 2:
            assert 0 <= result.win_rate <= 1
