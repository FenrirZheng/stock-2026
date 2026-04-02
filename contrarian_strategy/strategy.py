from dataclasses import dataclass
from typing import Optional


@dataclass
class TradeRecord:
    buy_date: str
    buy_price: float
    sell_date: Optional[str] = None
    sell_price: Optional[float] = None
    exit_type: Optional[str] = None  # "protection" / "ma_cross" / "end_of_data"
    return_pct: Optional[float] = None


def check_entry(close: float, ma_value: float, entry_distance: float) -> bool:
    """進場判斷：收盤價低於 MA × (1 - entry_distance/100) 時買進。"""
    return close < ma_value * (1 - entry_distance / 100)


def check_protection_exit(
    close: float,
    buy_price: float,
    protection_pct: float,
    protection_activated: bool,
) -> tuple[bool, bool]:
    """保護出場判斷。

    回傳 (should_exit, new_protection_activated)。
    先檢查是否達到保護門檻（啟動保護），再檢查是否跌破保護線。
    """
    threshold = buy_price * (1 + protection_pct / 100)

    if close >= threshold:
        protection_activated = True

    should_exit = protection_activated and close < threshold
    return should_exit, protection_activated


def check_ma_cross_exit(close: float, ma_value: float) -> bool:
    """均線交叉出場：收盤價回到 MA 上方時賣出。"""
    return close > ma_value
