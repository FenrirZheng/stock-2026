import pandas as pd
import yfinance as yf


def fetch_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """下載股票歷史資料，回傳含 OHLCV 的 DataFrame。"""
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker} from {start} to {end}")

    # yfinance 回傳 MultiIndex columns 時，展平它
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # 去除時區，簡化下游處理
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    return df


def add_moving_average(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """在 DataFrame 加入 MA 欄位。"""
    df = df.copy()
    df["MA"] = df["Close"].rolling(period).mean()
    return df
