from boolean_channel.strategy import check_entry, check_sma_exit, check_stop_loss


class TestCheckEntry:
    def test_close_below_lower(self):
        assert check_entry(close=85.0, lower_band=90.0) is True

    def test_close_equal_lower(self):
        # close ≤ lower → 進場
        assert check_entry(close=90.0, lower_band=90.0) is True

    def test_close_above_lower(self):
        assert check_entry(close=91.0, lower_band=90.0) is False


class TestCheckSmaExit:
    def test_close_below_sma(self):
        # close < SMA → 出場
        assert check_sma_exit(close=99.0, sma=100.0) is True

    def test_close_equal_sma(self):
        # close == SMA → 不出場（需嚴格跌破）
        assert check_sma_exit(close=100.0, sma=100.0) is False

    def test_close_above_sma(self):
        assert check_sma_exit(close=101.0, sma=100.0) is False


class TestCheckStopLoss:
    def test_triggered(self):
        # entry=100, stop=2% → threshold=98, close=97 → 停損
        assert check_stop_loss(close=97.0, entry_price=100.0, stop_pct=0.02) is True

    def test_boundary(self):
        # close 剛好 = threshold → 停損
        assert check_stop_loss(close=98.0, entry_price=100.0, stop_pct=0.02) is True

    def test_not_triggered(self):
        assert check_stop_loss(close=99.0, entry_price=100.0, stop_pct=0.02) is False
