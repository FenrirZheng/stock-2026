# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python 股票逆勢策略回測 (contrarian stock trading strategy backtesting) 專案。
策略邏輯：當股價跌破均線一定距離時進場，回到均線上方或觸發保護機制時出場。
使用 Optuna Bayesian Optimization 搜尋最佳參數組合 (MA 週期、進場距離、保護百分比)。

## Language & Conventions

- 語言：Python


## Commands

```bash
# 安裝依賴
source venv/bin/activate
pip install -r requirements.txt

# 執行回測
python -m contrarian_strategy.main

# 執行測試
pytest contrarian_strategy/tests/

# 執行單一測試
pytest contrarian_strategy/tests/test_xxx.py::test_function_name -v
```

## Architecture

```
contrarian_strategy/       # 主要 package（所有模組使用相對 import）
├── config.py              # 回測參數與搜尋範圍
├── data_fetcher.py        # yfinance 資料下載 + MA 計算
├── strategy.py            # 進出場判斷邏輯
├── backtest.py            # 回測引擎
├── optimizer.py           # Optuna 參數優化
├── reporter.py            # 結果輸出與當前訊號判斷
├── main.py                # 入口點
└── tests/
    ├── test_backtest.py
    └── test_strategy.py
```

## max_sharpe_ma

Bayesian Optimization 搜尋最佳 SMA 週期 (maximize Sharpe Ratio)。

```bash
# 執行分析
python -m max_sharpe_ma.main

# 執行測試
pytest max_sharpe_ma/tests/

# 設定標的與參數
# 編輯 max_sharpe_ma/config.py
```

```
max_sharpe_ma/             # MA 週期 Sharpe 最大化搜尋
├── config.py              # 標的、日期、搜尋範圍、Optuna 參數
├── data_fetcher.py        # yfinance 下載 + SMA + train/test split
├── strategy.py            # 持倉判斷（close > SMA → 持有）
├── backtest.py            # 回測引擎（Sharpe Ratio）
├── optimizer.py           # Optuna BO + Brute-Force 搜尋
├── reporter.py            # 比較報告 + 過擬合檢查
├── main.py                # 入口點
└── tests/
    ├── test_strategy.py
    └── test_backtest.py
```

## Gotchas

- Package 內使用相對 import，不能直接 `python contrarian_strategy/main.py`，必須用 `python -m`
