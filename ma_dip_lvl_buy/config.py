TICKER = "6854.TW"
START_DATE = "2022-08-01"
END_DATE = "2026-04-03"

TRAIN_RATIO = 0.7

# --- Optuna Optimization ---
N_TRIALS = 200
N_STARTUP_TRIALS = 30
RANDOM_STATE = 42

# --- 參數搜尋範圍 ---
PARAM_RANGES = {
    "ma_period": (10, 120),         # MA 週期
    "dip_pct": (1.0, 30.0),        # 買進觸發：股價低於均線 dip_pct%
    "rsi_threshold": (15, 50),      # RSI 低於此值才允許進場
    "timeout_days": (3, 60),        # 停損觸發：未突破均線最大天數
    "hard_stop_pct": (5.0, 40.0),  # 固定止損：從買入價跌 hard_stop_pct%
    "trail_stop_pct": (2.0, 20.0), # Trailing Stop：從波段高點回落 trail_stop_pct%
}

# --- RSI ---
RSI_PERIOD = 14

# --- 均線突破確認 ---
MA_CROSS_CONFIRM_DAYS = 2  # 連續幾天站上均線才算突破

# --- 交易成本（台股） ---
COMMISSION_RATE = 0.001425   # 手續費 0.1425%（買賣各一次）
TAX_RATE = 0.003             # 證交稅 0.3%（賣出）
SLIPPAGE_RATE = 0.001        # 滑價估計 0.1%（單邊）

# --- 懲罰條件 ---
MIN_TRADES = 10
MAX_DRAWDOWN_LIMIT = 0.40
