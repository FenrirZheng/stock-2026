# Bayesian Optimization 股票策略參數搜尋器 — 設計規格

## Context

量化交易策略的核心瓶頸是參數調優。當可調參數超過 2 維時，Grid Search 因維度詛咒不再可行。本專案使用 Bayesian Optimization (Optuna TPE) 搜尋 3 維參數空間，針對台股鴻海 (2317.TW) 找出最佳交易策略參數。

### 求解目標

給定策略參數 (x₁, x₂, x₃)，最大化累積報酬率 w = ∏(1+rᵢ) - 1。

## 決策紀錄

| 項目 | 選擇 | 理由 |
|------|------|------|
| 資料來源 | yfinance (2317.TW) | 免費、支援台股、API 簡單 |
| BO 框架 | Optuna (TPE sampler) | 內建 fANOVA、視覺化、API 簡潔 |
| 回測期間 | 10 年 (2016~2026) | 涵蓋多種市況，樣本充足 |
| 目標函數 | 累積報酬率 | 直接對應投資目標 |
| 出場機制 | 保護出場 + 均線交叉出場 | 雙重保障，避免套牢 |
| 架構 | 模組化多檔 | 可測試、可擴展、依賴輕量 |

## 策略邏輯

### 三個參數

| 參數 | 符號 | 範圍 | 型別 | 說明 |
|------|------|------|------|------|
| MA 週期 | x₁ | 5 ~ 240 | int | 移動平均天數 |
| 進場距離 | x₂ | 1.0 ~ 30.0 | float | 收盤價低於 MA 幾 % 時進場 |
| 保護百分比 | x₃ | 1.0 ~ 15.0 | float | 獲利保護門檻 |

### 狀態機

```
State: IDLE（空手）
  每天計算 MA(x₁)
  若 close < MA × (1 - x₂/100) → BUY at close，進入 HOLDING
    紀錄 buy_price = close
    protection_activated = False

State: HOLDING（持有）
  每天依序檢查：
  1. 若 close ≥ buy_price × (1 + x₃/100) → protection_activated = True
  2. 出場條件（依序檢查，觸發任一即賣出）：
     a. 保護出場：protection_activated AND close < buy_price × (1 + x₃/100) → SELL
     b. 均線交叉出場：close > MA(x₁) → SELL
  3. 都未觸發 → 繼續持有

  回測結束若仍持有 → 以最後收盤價強制平倉，exit_type = "end_of_data"
```

### 報酬計算

每筆交易報酬 rᵢ = (sell_price - buy_price) / buy_price
累積報酬 w = ∏(1 + rᵢ) - 1

## 架構

### 檔案結構

```
stock/
  main.py              # 進入點，串接所有模組（~40 行）
  config.py            # 常數、參數範圍（~30 行）
  data_fetcher.py      # yfinance 包裝、MA 計算（~50 行）
  strategy.py          # 純函數：進場/出場邏輯判斷（~80 行）
  backtest.py          # 回測引擎：逐列迭代套用策略（~100 行）
  optimizer.py         # Optuna study、objective、fANOVA（~80 行）
  reporter.py          # 報表輸出、當前訊號判斷（~100 行）
  requirements.txt     # 依賴清單
  tests/
    __init__.py
    test_strategy.py   # 策略純函數單元測試
    test_backtest.py   # 合成資料整合測試
```

### 模組職責

**config.py**
- `TICKER`, `START_DATE`, `END_DATE`
- 參數範圍常數：`MA_RANGE`, `ENTRY_DISTANCE_RANGE`, `PROTECTION_RANGE`
- `N_TRIALS = 200`

**data_fetcher.py**
- `fetch_stock_data(ticker, start, end) -> pd.DataFrame`：yfinance 下載，驗證欄位，去除時區
- `add_moving_average(df, period) -> pd.DataFrame`：加入 MA 欄位

**strategy.py**（核心——純函數，無副作用）
- `TradeRecord` dataclass：buy_date, buy_price, sell_date, sell_price, exit_type, return_pct
- `check_entry(close, ma_value, entry_distance) -> bool`
- `check_protection_exit(close, buy_price, protection_pct, protection_activated) -> tuple[bool, bool]`
- `check_ma_cross_exit(close, ma_value) -> bool`

**backtest.py**
- `BacktestResult` dataclass：trades, cumulative_return, trade_count, win_rate, exit_type_counts
- `run_backtest(df, ma_period, entry_distance, protection_pct) -> BacktestResult`

**optimizer.py**
- `create_objective(df) -> Callable`：回傳 Optuna objective（DataFrame 在 closure 中）
- `run_optimization(df, n_trials, seed) -> optuna.Study`
- `get_parameter_importance(study) -> dict`：fANOVA 參數重要性

**reporter.py**
- `print_optimization_summary(study)`：最佳參數、最佳值、參數重要性排序
- `print_trade_statistics(result: BacktestResult)`：勝率、交易次數、平均盈虧、出場類型分佈
- `generate_current_signal(df, best_params) -> str`：用最佳參數判斷目前訊號（BUY/SELL/WAIT）
- `plot_optimization_history(study)`：優化過程圖
- `plot_parameter_importance(study)`：參數重要性柱狀圖

**main.py**
```python
def main():
    df = fetch_stock_data(TICKER, START_DATE, END_DATE)
    study = run_optimization(df, N_TRIALS)
    best = study.best_params
    result = run_backtest(df, best['ma_period'], best['entry_distance'], best['protection_pct'])
    print_optimization_summary(study)
    print_trade_statistics(result)
    signal = generate_current_signal(df, best)
```

### 資料流

```
main.py
  → data_fetcher.fetch_stock_data("2317.TW") → DataFrame (2429 rows, 10yr)
  → optimizer.run_optimization(df, 200 trials)
      → [每次 trial]
          trial.suggest_int("ma_period", 5, 240)
          trial.suggest_float("entry_distance", 1.0, 30.0)
          trial.suggest_float("protection_pct", 1.0, 15.0)
          → backtest.run_backtest(df, x₁, x₂, x₃)
              → strategy.check_entry()
              → strategy.check_protection_exit()
              → strategy.check_ma_cross_exit()
              → BacktestResult
          → return cumulative_return
  → reporter.print_optimization_summary(study)
  → reporter.print_trade_statistics(best_result)
  → reporter.generate_current_signal(df, best_params)
```

## 輸出格式

### 優化摘要
```
══════════════════════════════════════════
  Bayesian Optimization 結果 — 2317.TW 鴻海
══════════════════════════════════════════
  最佳參數：
    MA 週期 (x₁)     : 10 天
    進場距離 (x₂)     : 14.92%
    保護百分比 (x₃)    : 4.41%
  最大累積報酬率      : +106.48%
  
  參數重要性 (fANOVA)：
    x₁ MA 週期        : ████████████████████░░  72%
    x₂ 進場距離       : █████░░░░░░░░░░░░░░░░  18%
    x₃ 保護百分比     : ██░░░░░░░░░░░░░░░░░░░  10%
══════════════════════════════════════════
```

### 交易統計
```
  交易次數: 42
  勝率    : 64.3%  (27 勝 / 15 負)
  平均獲利: +8.2%
  平均虧損: -3.1%
  盈虧比  : 2.65

  出場類型分佈：
    保護出場      : 18 (42.9%)
    均線交叉出場  : 22 (52.4%)
    回測結束平倉  : 2  (4.8%)
```

### 當前訊號
```
  ── 當前市場訊號 ──
  日期: 2026-04-01
  收盤: 178.5
  MA(10): 182.3
  距離 MA: -2.09%
  進場門檻: -14.92%
  
  訊號: ⏳ 觀望
  原因: 目前距離 MA 僅 -2.09%，尚未達到進場門檻 -14.92%
```

## 依賴

```
yfinance>=0.2.0
optuna>=3.0.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
```

## 測試策略

### test_strategy.py
- 測試 `check_entry`：close 在 MA 下方不同距離 → 預期 True/False
- 測試 `check_protection_exit`：各種 close/buy_price/protection_activated 組合
- 測試 `check_ma_cross_exit`：close 在 MA 上下方

### test_backtest.py
- 合成 20 筆 DataFrame，手動計算預期交易結果
- 驗證 cumulative_return 計算正確性
- 驗證回測結束強制平倉邏輯

## 驗證計畫

1. `pip install -r requirements.txt` 安裝依賴
2. `pytest tests/ -v` 單元測試全過
3. `python main.py` 完整執行，確認：
   - yfinance 成功下載 2317.TW 資料
   - Optuna 完成 200 次試驗
   - 輸出最佳參數、累積報酬率、fANOVA 重要性
   - 輸出交易統計（勝率、交易次數等）
   - 輸出當前市場訊號
4. 檢查結果合理性：累積報酬率在合理範圍、交易次數 > 10
