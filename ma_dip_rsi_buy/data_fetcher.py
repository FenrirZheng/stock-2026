import pandas as pd
import yfinance as yf


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


def add_sma(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """在 DataFrame 加入 SMA 欄位，並移除 NaN 列。"""
    df = df.copy()
    df["SMA"] = df["Close"].rolling(period).mean()
    df = df.dropna(subset=["SMA"])
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """在 DataFrame 加入 RSI 欄位，並移除 NaN 列。"""
    df = df.copy()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - 100 / (1 + rs)
    df = df.dropna(subset=["RSI"])
    return df


def split_train_test(
    df: pd.DataFrame, split_date: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """依固定日期切分訓練集與測試集。"""
    return df.loc[:split_date].copy(), df.loc[split_date:].iloc[1:].copy()
