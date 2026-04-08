TICKER = "3711.TW"
START_DATE = "2015-01-01"
END_DATE = "2026-04-03"

TRAIN_RATIO = 0.8

# --- Bayesian Optimization ---
INIT_POINTS = 30
N_ITER = 150
RANDOM_STATE = 42

# --- 參數搜尋範圍 ---
PBOUNDS = {
    "x": (5, 120),      # MA 週期（擴大到 120）
    "m": (1.0, 30.0),   # 買進觸發：股價低於均線 m%（擴大到 30%）
    "n": (3, 60),        # 停損觸發：未突破均線最大天數（擴大到 60）
    "k": (5.0, 40.0),   # 固定止損：從買入價跌 k%（擴大到 40%）
    "t": (2.0, 20.0),   # Trailing Stop：從波段高點回落 t%（擴大到 20%）
    "rsi_threshold": (15, 50),  # RSI 低於此值才允許進場
}

# --- RSI ---
RSI_PERIOD = 14

# --- 交易成本（台股） ---
COMMISSION_RATE = 0.001425   # 手續費 0.1425%（買賣各一次）
TAX_RATE = 0.003             # 證交稅 0.3%（賣出）

# --- 懲罰條件 ---
MIN_TRADES = 10
MAX_DRAWDOWN_LIMIT = 0.40
PENALTY_SCORE = -1.0
