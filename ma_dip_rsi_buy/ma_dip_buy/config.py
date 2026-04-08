TICKER = "2330.TW"
START_DATE = "2015-01-01"
END_DATE = "2026-04-01"

TRAIN_RATIO = 0.8

# --- Bayesian Optimization ---
INIT_POINTS = 20
N_ITER = 100
RANDOM_STATE = 42

# --- 參數搜尋範圍 ---
PBOUNDS = {
    "x": (5, 59),       # MA 週期
    "m": (1.0, 15.0),   # 買進觸發：股價低於均線 m%
    "n": (3, 30),        # 停損觸發：未突破均線最大天數
    "k": (3.0, 20.0),   # 固定止損：從買入價跌 k%
    "t": (2.0, 15.0),   # Trailing Stop：從波段高點回落 t%
}

# --- 交易成本（台股） ---
COMMISSION_RATE = 0.001425   # 手續費 0.1425%（買賣各一次）
TAX_RATE = 0.003             # 證交稅 0.3%（賣出）

# --- 懲罰條件 ---
MIN_TRADES = 10
MAX_DRAWDOWN_LIMIT = 0.40
PENALTY_SCORE = -1.0
