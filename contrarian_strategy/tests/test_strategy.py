from ..strategy import check_entry, check_protection_exit, check_ma_cross_exit


class TestCheckEntry:
    def test_below_threshold_triggers_entry(self):
        # close=85, MA=100, distance=10% → threshold=90 → 85 < 90 → True
        assert check_entry(85, 100, 10) is True

    def test_above_threshold_no_entry(self):
        # close=95, MA=100, distance=10% → threshold=90 → 95 < 90 → False
        assert check_entry(95, 100, 10) is False

    def test_exactly_at_threshold_no_entry(self):
        # close=90, MA=100, distance=10% → threshold=90 → 90 < 90 → False
        assert check_entry(90, 100, 10) is False

    def test_large_distance(self):
        # close=72, MA=100, distance=30% → threshold=70 → 72 < 70 → False
        assert check_entry(72, 100, 30) is False
        # close=69 → True
        assert check_entry(69, 100, 30) is True


class TestCheckProtectionExit:
    def test_not_activated_below_threshold(self):
        # buy=100, prot=5% → threshold=105, close=103 → 不啟動、不出場
        should_exit, activated = check_protection_exit(103, 100, 5, False)
        assert should_exit is False
        assert activated is False

    def test_activate_when_reaching_threshold(self):
        # close=106 ≥ 105 → 啟動，但不出場（因為沒跌破）
        should_exit, activated = check_protection_exit(106, 100, 5, False)
        assert should_exit is False
        assert activated is True

    def test_exit_after_activation_and_drop(self):
        # 已啟動，close=104 < 105 → 出場
        should_exit, activated = check_protection_exit(104, 100, 5, True)
        assert should_exit is True
        assert activated is True

    def test_no_exit_when_activated_but_still_above(self):
        # 已啟動，close=107 ≥ 105 → 繼續持有
        should_exit, activated = check_protection_exit(107, 100, 5, True)
        assert should_exit is False
        assert activated is True

    def test_exactly_at_threshold_activates_but_no_exit(self):
        # close=105 == threshold → 啟動保護，但 close < threshold 不成立 → 不出場
        should_exit, activated = check_protection_exit(105, 100, 5, False)
        assert should_exit is False
        assert activated is True

    def test_same_day_activation_and_drop_impossible(self):
        # 同一天 close 只有一個值，若 close ≥ threshold 則啟動但不出場
        # 若 close < threshold 且未啟動，則不出場
        should_exit, activated = check_protection_exit(104, 100, 5, False)
        assert should_exit is False
        assert activated is False


class TestCheckMaCrossExit:
    def test_above_ma_triggers_exit(self):
        assert check_ma_cross_exit(105, 100) is True

    def test_below_ma_no_exit(self):
        assert check_ma_cross_exit(95, 100) is False

    def test_exactly_at_ma_no_exit(self):
        # close == MA → close > MA 不成立 → 不出場
        assert check_ma_cross_exit(100, 100) is False
