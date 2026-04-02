# Stock — 股票策略回測工具集

使用 Optuna Bayesian Optimization 搜尋最佳參數的股票策略回測專案。每個策略為獨立 package。

## 策略工具

### contrarian_strategy — 逆勢策略回測

當股價跌破均線一定距離時進場，回到均線上方或觸發保護機制時出場。

- **進場**：收盤價 < MA × (1 - entry_distance%)
- **出場**：均線交叉（收盤 > MA）或保護出場（漲到門檻後回跌）
- **優化參數**：MA 週期 (5–240)、進場距離 (1–30%)、保護百分比 (1–15%)
- **預設標的**：^TWII（加權指數）

```bash
python -m contrarian_strategy.main
```

### max_sharpe_ma — 最佳 SMA 週期搜尋

搜尋使 Sharpe Ratio 最大化的 SMA 週期。close > SMA 時持有，否則空手。

- **持倉邏輯**：position[t-1] 決定 day t 報酬，避免前視偏差
- **優化方式**：Bayesian Optimization + Brute-Force 雙重驗證
- **過擬合檢查**：70/30 train/test split，比較 baseline MA(20)
- **預設標的**：00635U.TW（元大S&P原油正2）

```bash
python -m max_sharpe_ma.main
```

### bollinger_contrarian — 布林通道逆勢策略

跌到下軌買進的逆勢邏輯：股價觸及布林通道下軌時視為超賣進場，回到中線（SMA）或觸發停損時出場。

- **進場**：收盤價 ≤ 下軌（SMA − N × σ）
- **出場**：SMA 跌破（收盤 < SMA）或停損（跌幅達 5%）
- **優化參數**：MA 週期 (20–200)、標準差倍數 (1.0–3.0)
- **優化目標**：WinRate − λ · max(0, −TotalReturn)，含硬約束（最少 15 筆交易、最大回撤 35%）
- **預設標的**：00631L.TW（元大台灣50正2）

```bash
python -m bollinger_contrarian.main
```

## 快速開始

```bash
source venv/bin/activate
pip install -r requirements.txt

# 執行任一策略
python -m contrarian_strategy.main
python -m max_sharpe_ma.main

# 執行測試
pytest
```

## 依賴

- Python 3.11
- yfinance — 股價資料下載
- optuna — Bayesian Optimization
- pandas / numpy — 資料處理
- matplotlib — 圖表輸出
- scikit-learn — 統計工具
