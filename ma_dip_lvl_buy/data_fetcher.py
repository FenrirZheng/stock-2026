import pandas as pd
import yfinance as yf

from .config import RSI_PERIOD


def fetch_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """下載股票歷史資料，回傳含 OHLCV 的 DataFrame。"""
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker} from {start} to {end}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    return df


def add_indicators(df: pd.DataFrame, ma_period: int) -> pd.DataFrame:
    """加入 SMA 與 RSI 指標，移除前段 NaN 列。"""
    df = df.copy()
    df["SMA"] = df["Close"].rolling(ma_period).mean()
    df["RSI"] = _compute_rsi(df["Close"], RSI_PERIOD)
    df = df.dropna(subset=["SMA", "RSI"])
    return df


def _compute_rsi(series: pd.Series, period: int) -> pd.Series:
    """用 EMA 平滑法計算 RSI。"""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def split_train_test(
    df: pd.DataFrame, ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """依比例切分訓練集與測試集。"""
    split_idx = int(len(df) * ratio)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
