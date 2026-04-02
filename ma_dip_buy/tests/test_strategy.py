import pandas as pd
import pytest

from ma_dip_buy.strategy import run_trades


def _make_df(closes: list[float], sma: list[float]) -> pd.DataFrame:
    """建立測試用 DataFrame，含 Close 和 SMA 欄位。"""
    return pd.DataFrame({"Close": closes, "SMA": sma})


class TestEntry:
    def test_entry_when_close_below_threshold(self):
        """股價低於 SMA × (1 - m/100) 時進場。"""
        # SMA = 100, m = 10 → threshold = 90
        # Day 0: Close=95 (above 90, no entry)
        # Day 1: Close=89 (below 90, entry)
        # Day 2: Close=80 (hard stop: 89 * 0.85 = 75.65, no)
        # Day 3: Close=70 (hard stop: 89 * 0.85 = 75.65, yes → exit)
        closes = [95, 89, 80, 70]
        sma = [100, 100, 100, 100]
        trades = run_trades(_make_df(closes, sma), x=5, m=10, n=30, k=15, t=10)
        assert len(trades) == 1
        assert trades[0].entry_price == 89
        assert trades[0].entry_day == 1

    def test_no_entry_when_above_threshold(self):
        """股價在均線上方不進場。"""
        closes = [105, 102, 108]
        sma = [100, 100, 100]
        trades = run_trades(_make_df(closes, sma), x=5, m=10, n=30, k=15, t=10)
        assert len(trades) == 0


class TestHardStop:
    def test_hard_stop_exit(self):
        """固定止損：Close < entry_price × (1 - k/100)。"""
        # m=6 → threshold=100*0.94=94. Day1: close=90 < 94 → entry at 90
        # k=10 → hard stop at 81. t=15 → trailing stop at 90*0.85=76.5
        # Day3: close=80 < 81 → hard stop (but 80 > 76.5, trailing doesn't fire)
        closes = [95, 90, 85, 80]
        sma = [100, 100, 100, 100]
        trades = run_trades(_make_df(closes, sma), x=5, m=6, n=30, k=10, t=15)
        assert len(trades) == 1
        assert trades[0].entry_price == 90
        assert trades[0].exit_reason == "hard_stop"
        assert trades[0].exit_price == 80


class TestTrailingStop:
    def test_trailing_stop_exit(self):
        """Trailing Stop：Close < peak × (1 - t/100)。"""
        # entry at 89, peak goes to 95, t=5 → stop at 90.25
        # Day 4: close=90 < 90.25 → trailing stop
        closes = [95, 89, 92, 95, 90]
        sma = [100, 100, 100, 100, 100]
        trades = run_trades(_make_df(closes, sma), x=5, m=10, n=30, k=15, t=5)
        assert len(trades) == 1
        assert trades[0].exit_reason == "trailing_stop"


class TestTimeout:
    def test_timeout_exit(self):
        """超時出場：持倉 n 天且從未 Close >= SMA。"""
        # entry at 89, n=3, never crosses SMA=100
        closes = [95, 89, 88, 87, 88]
        sma = [100, 100, 100, 100, 100]
        trades = run_trades(_make_df(closes, sma), x=5, m=10, n=3, k=20, t=15)
        assert len(trades) == 1
        assert trades[0].exit_reason == "timeout"
        # days held = 4-1 = 3, which is >= n=3
        assert trades[0].exit_day - trades[0].entry_day >= 3


class TestMACrossback:
    def test_ma_crossback_exit(self):
        """突破後跌破均線出場。"""
        # entry at 89, crosses SMA at day 2 (close=101>=100)
        # day 3: close=98 < SMA=100 → crossback exit
        closes = [95, 89, 101, 98]
        sma = [100, 100, 100, 100]
        trades = run_trades(_make_df(closes, sma), x=5, m=10, n=30, k=20, t=15)
        assert len(trades) == 1
        assert trades[0].exit_reason == "ma_crossback"


class TestMultipleTrades:
    def test_multiple_trades(self):
        """多筆交易：出場後重新進場。"""
        closes = [95, 89, 101, 98, 95, 88, 102, 99]
        sma = [100, 100, 100, 100, 100, 100, 100, 100]
        trades = run_trades(_make_df(closes, sma), x=5, m=10, n=30, k=20, t=15)
        assert len(trades) == 2
        assert trades[0].exit_reason == "ma_crossback"
        assert trades[1].exit_reason == "ma_crossback"
