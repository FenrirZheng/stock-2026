import numpy as np
import pandas as pd
import pytest

from ma_dip_lvl_buy.strategy import Trade, run_trades


def _make_df(closes, opens=None, sma=None, rsi=None, start="2024-01-02"):
    """建立測試用 DataFrame，含 Close/Open/SMA/RSI。"""
    n = len(closes)
    dates = pd.bdate_range(start, periods=n)
    if opens is None:
        opens = closes
    if sma is None:
        sma = [100.0] * n
    if rsi is None:
        rsi = [30.0] * n
    df = pd.DataFrame({
        "Close": closes,
        "Open": opens,
        "High": closes,
        "Low": closes,
        "SMA": sma,
        "RSI": rsi,
    }, index=dates)
    return df


class TestEntrySignal:
    """進場訊號測試。"""

    def test_entry_on_next_day_open(self):
        """訊號日隔天以 Open 進場。"""
        # Day 0: Close=80 < SMA=100*(1-5/100)=95, RSI=25 < 30 → signal
        # Day 1: Open=82 → entry
        # Day 2-9: hold, no exit triggered
        closes = [80.0] + [90.0] * 9
        opens = [80.0, 82.0] + [90.0] * 8
        sma = [100.0] * 10
        rsi = [25.0] + [50.0] * 9  # RSI only matters on signal day

        df = _make_df(closes, opens, sma, rsi)
        trades = run_trades(df, ma_period=20, dip_pct=5.0,
                            rsi_threshold=30.0, timeout_days=60,
                            hard_stop_pct=30.0, trail_stop_pct=15.0)

        assert len(trades) >= 1
        assert trades[0].entry_price == 82.0
        assert trades[0].signal_date == str(df.index[0].date())
        assert trades[0].entry_date == str(df.index[1].date())

    def test_no_entry_when_rsi_too_high(self):
        """RSI 高於門檻不進場。"""
        closes = [80.0] * 10
        sma = [100.0] * 10
        rsi = [60.0] * 10  # RSI > 30

        df = _make_df(closes, sma=sma, rsi=rsi)
        trades = run_trades(df, ma_period=20, dip_pct=5.0,
                            rsi_threshold=30.0, timeout_days=60,
                            hard_stop_pct=30.0, trail_stop_pct=15.0)

        assert len(trades) == 0

    def test_no_entry_when_dip_not_enough(self):
        """跌幅不夠不進場。"""
        # dip_pct=5 → need Close < 100*0.95=95. Close=96 doesn't qualify
        closes = [96.0] * 10
        sma = [100.0] * 10
        rsi = [25.0] * 10

        df = _make_df(closes, sma=sma, rsi=rsi)
        trades = run_trades(df, ma_period=20, dip_pct=5.0,
                            rsi_threshold=30.0, timeout_days=60,
                            hard_stop_pct=30.0, trail_stop_pct=15.0)

        assert len(trades) == 0


class TestExitConditions:
    """出場條件測試。"""

    def test_hard_stop(self):
        """固定止損觸發。"""
        # Day 0: signal (Close=80, SMA=100, RSI=25)
        # Day 1: entry at Open=82
        # Day 2: Close=70 < 82*(1-10/100)=73.8 → hard_stop
        closes = [80.0, 82.0, 70.0] + [70.0] * 7
        opens = [80.0, 82.0, 75.0] + [70.0] * 7
        sma = [100.0] * 10
        rsi = [25.0] + [50.0] * 9

        df = _make_df(closes, opens, sma, rsi)
        trades = run_trades(df, ma_period=20, dip_pct=5.0,
                            rsi_threshold=30.0, timeout_days=60,
                            hard_stop_pct=10.0, trail_stop_pct=5.0)

        assert len(trades) >= 1
        assert trades[0].exit_reason == "hard_stop"
        assert trades[0].exit_price == 70.0

    def test_trailing_stop(self):
        """Trailing stop 觸發。"""
        # Day 0: signal (Close=80)
        # Day 1: entry at Open=82
        # Day 2: Close=95 → peak=95
        # Day 3: Close=89 < 95*(1-5/100)=90.25 → trailing_stop
        closes = [80.0, 82.0, 95.0, 89.0] + [89.0] * 6
        opens = [80.0, 82.0, 85.0, 93.0] + [89.0] * 6
        sma = [100.0] * 10
        rsi = [25.0] + [50.0] * 9

        df = _make_df(closes, opens, sma, rsi)
        trades = run_trades(df, ma_period=20, dip_pct=5.0,
                            rsi_threshold=30.0, timeout_days=60,
                            hard_stop_pct=30.0, trail_stop_pct=5.0)

        assert len(trades) >= 1
        assert trades[0].exit_reason == "trailing_stop"

    def test_timeout(self):
        """超時未突破均線觸發。"""
        # Entry at Day 1, all closes stay below SMA, timeout_days=3
        closes = [80.0, 82.0, 83.0, 84.0, 85.0] + [85.0] * 5
        opens = [80.0, 82.0, 83.0, 84.0, 85.0] + [85.0] * 5
        sma = [100.0] * 10  # always above
        rsi = [25.0] + [50.0] * 9

        df = _make_df(closes, opens, sma, rsi)
        trades = run_trades(df, ma_period=20, dip_pct=5.0,
                            rsi_threshold=30.0, timeout_days=3,
                            hard_stop_pct=30.0, trail_stop_pct=15.0)

        assert len(trades) >= 1
        assert trades[0].exit_reason == "timeout"

    def test_end_of_data_closes_open_position(self):
        """回測結束仍持倉 → 強制平倉。"""
        # Day 0: signal, Day 1: entry, no exit triggers in remaining days
        closes = [80.0, 82.0, 83.0, 84.0, 85.0]
        opens = [80.0, 82.0, 83.0, 84.0, 85.0]
        sma = [100.0] * 5  # always above, no crossback possible
        rsi = [25.0] + [50.0] * 4

        df = _make_df(closes, opens, sma, rsi)
        trades = run_trades(df, ma_period=20, dip_pct=5.0,
                            rsi_threshold=30.0, timeout_days=60,
                            hard_stop_pct=30.0, trail_stop_pct=15.0)

        assert len(trades) == 1
        assert trades[0].exit_reason == "end_of_data"
        assert trades[0].exit_price == 85.0


class TestMaCrossback:
    """均線跌破出場測試。"""

    def test_ma_crossback_after_confirmed_cross(self):
        """先連續站上均線確認，再跌破 → ma_crossback。"""
        # Day 0: signal (Close=80 < SMA=100*0.95)
        # Day 1: entry at Open=82
        # Day 2: Close=101 > SMA=100 (consecutive_above=1)
        # Day 3: Close=102 > SMA=100 (consecutive_above=2 → confirmed)
        # Day 4: Close=99 < SMA=100 → ma_crossback
        closes = [80.0, 82.0, 101.0, 102.0, 99.0] + [99.0] * 5
        opens = [80.0, 82.0, 90.0, 101.0, 102.0] + [99.0] * 5
        sma = [100.0] * 10
        rsi = [25.0] + [50.0] * 9

        df = _make_df(closes, opens, sma, rsi)
        trades = run_trades(df, ma_period=20, dip_pct=5.0,
                            rsi_threshold=30.0, timeout_days=60,
                            hard_stop_pct=30.0, trail_stop_pct=15.0)

        assert len(trades) >= 1
        assert trades[0].exit_reason == "ma_crossback"
