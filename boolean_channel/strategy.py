from dataclasses import dataclass
from typing import Optional


@dataclass
class TradeRecord:
    signal_date: str
    buy_date: str
    buy_price: float
    sell_date: Optional[str] = None
    sell_price: Optional[float] = None
    exit_type: Optional[str] = None  # "sma_cross" / "stop_loss" / "end_of_data"
    pnl: Optional[float] = None


def check_entry(close: float, lower_band: float) -> bool:
    """進場判斷：收盤價 ≤ 下軌時觸發買進信號。"""
    return close <= lower_band


def check_sma_exit(close: float, sma: float) -> bool:
    """中線出場：收盤價跌破 SMA 時觸發賣出信號。"""
    return close < sma


def check_stop_loss(close: float, entry_price: float, stop_pct: float) -> bool:
    """停損判斷：收盤價 ≤ 進場價 × (1 - s) 時觸發。"""
    return close <= entry_price * (1 - stop_pct)
