import pandas as pd

from ..backtest import run_backtest


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


class TestRunBacktest:
    def test_single_trade_protection_exit(self):
        """價格下跌進場 → 上漲觸發保護 → 回落觸發保護出場。"""
        # MA(3) 前 3 天建立均線，然後觸發進場和出場
        closes = [
            100, 100, 100,  # MA=100
            85,              # close=85 < 100*(1-10/100)=90 → BUY at 85
            92,              # 85*1.05=89.25, 92≥89.25 → protection ON, 92>MA? MA≈(100+100+85)/3=95 → 92<95 no
            88,              # 88<89.25 且 protection ON → SELL at 88
        ]
        df = _make_df(closes)
        result = run_backtest(df, ma_period=3, entry_distance=10, protection_pct=5)

        assert result.trade_count == 1
        t = result.trades[0]
        assert t.buy_price == 85
        assert t.sell_price == 88
        assert t.exit_type == "protection"
        assert abs(t.return_pct - (88 - 85) / 85) < 1e-9

    def test_ma_cross_exit(self):
        """價格跌破進場 → 反彈回 MA 上方 → 均線交叉出場。"""
        closes = [
            100, 100, 100,   # MA=100
            85,               # BUY at 85 (85<90)
            88,               # 85*1.05=89.25, 88<89.25 → protection OFF, MA≈(100+85+88)/3≈91, 88<91 no
            102,              # protection: 102≥89.25 → ON, 102<89.25? no. MA≈(85+88+102)/3=91.67, 102>91.67 → ma_cross SELL
        ]
        df = _make_df(closes)
        result = run_backtest(df, ma_period=3, entry_distance=10, protection_pct=5)

        assert result.trade_count == 1
        assert result.trades[0].exit_type == "ma_cross"
        assert result.trades[0].sell_price == 102

    def test_end_of_data_exit(self):
        """回測結束仍持有 → 強制平倉。"""
        closes = [
            100, 100, 100,  # MA=100
            85,              # BUY at 85
            86,              # 持有中，無出場條件
        ]
        df = _make_df(closes)
        result = run_backtest(df, ma_period=3, entry_distance=10, protection_pct=5)

        assert result.trade_count == 1
        assert result.trades[0].exit_type == "end_of_data"
        assert result.trades[0].sell_price == 86

    def test_no_trades_when_never_below_threshold(self):
        """價格始終高於進場門檻 → 不交易。"""
        closes = [100, 101, 102, 103, 104]
        df = _make_df(closes)
        result = run_backtest(df, ma_period=3, entry_distance=10, protection_pct=5)

        assert result.trade_count == 0
        assert result.cumulative_return == 0.0

    def test_cumulative_return_calculation(self):
        """驗證多筆交易的累積報酬計算。"""
        # MA(3) 是 rolling window，需要考慮 MA 隨價格變動
        closes = [
            100, 100, 100,   # MA=100
            85,               # BUY#1 at 85 (85<90)
            110,              # MA=(100+85+110)/3=98.3, 110>98.3 → ma_cross SELL
            110, 110, 110,   # 重建穩定 MA=110
            88,               # MA=(110+110+88)/3=102.67, 88<102.67*0.9=92.4 → BUY#2
            120,              # MA=(110+88+120)/3=106, 120>106 → ma_cross SELL
        ]
        df = _make_df(closes)
        result = run_backtest(df, ma_period=3, entry_distance=10, protection_pct=5)

        assert result.trade_count == 2
        r1 = (110 - 85) / 85
        r2 = (120 - 88) / 88
        expected = (1 + r1) * (1 + r2) - 1
        assert abs(result.cumulative_return - expected) < 1e-9

    def test_win_rate(self):
        """驗證勝率計算。"""
        closes = [
            100, 100, 100,
            85,              # BUY at 85
            110,             # SELL at 110 (win: +29.4%)
            100, 100, 100,
            88,              # BUY at 88
            80,              # 持有
            75,              # end_of_data SELL at 75 (loss: -14.8%)
        ]
        df = _make_df(closes)
        result = run_backtest(df, ma_period=3, entry_distance=10, protection_pct=5)

        assert result.trade_count == 2
        assert result.win_rate == 0.5
