# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python 股票策略回測專案，包含多個獨立策略 package。
每個 package 獨立開發，不共用程式碼（即使邏輯類似也從頭寫）。
Python 3.11 · Optuna · yfinance · pandas


## Commands

```bash
# 安裝依賴
source .venv/bin/activate   # 或 venv/bin/activate
pip install -r requirements.txt

# 執行測試
pytest                                    # 全部測試
pytest contrarian_strategy/tests/         # 單一 package
pytest max_sharpe_ma/tests/
pytest bollinger_contrarian/tests/

# 執行單一測試
pytest contrarian_strategy/tests/test_xxx.py::test_function_name -v

# 執行回測
python -m contrarian_strategy.main
python -m max_sharpe_ma.main
python -m bollinger_contrarian.main
```

## Architecture

每個策略 package 結構一致：

```
<package>/
├── __init__.py       # package marker
├── config.py         # 參數設定
├── data_fetcher.py   # 資料下載 + 指標計算
├── strategy.py       # 進出場邏輯
├── backtest.py       # 回測引擎
├── optimizer.py      # Optuna 參數優化
├── reporter.py       # 結果輸出
├── main.py           # 入口點 (python -m <package>.main)
└── tests/
```

### contrarian_strategy

逆勢策略：股價跌破均線一定距離時進場，回到均線上方或觸發保護機制時出場。
搜尋最佳 MA 週期、進場距離、保護百分比。

### max_sharpe_ma

Bayesian Optimization 搜尋最佳 SMA 週期 (maximize Sharpe Ratio)。
close > SMA → 持有，含 train/test split + 過擬合檢查。

### bollinger_contrarian

布林通道策略：以 SMA ± N 倍標準差建立通道，搜尋最佳 MA 週期與標準差倍數。
含 stop-loss、train/test split。

## Gotchas

- Package 內使用相對 import，不能直接 `python contrarian_strategy/main.py`，必須用 `python -m`
- 6854.TW 最早交易資料為 2022-08-19（約該時掛牌上市），START_DATE 設更早也無資料
