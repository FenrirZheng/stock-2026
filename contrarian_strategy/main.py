from .backtest import run_backtest
from .config import END_DATE, N_TRIALS, START_DATE, TICKER
from .data_fetcher import fetch_stock_data
from .optimizer import get_parameter_importance, run_optimization
from .reporter import (
    generate_current_signal,
    print_optimization_summary,
    print_trade_statistics,
)


def main():
    print(f"下載 {TICKER} 歷史資料 ({START_DATE} ~ {END_DATE})...")
    df = fetch_stock_data(TICKER, START_DATE, END_DATE)
    print(f"取得 {len(df)} 筆交易日資料")

    print(f"\n開始 Bayesian Optimization ({N_TRIALS} trials)...")
    study = run_optimization(df, N_TRIALS)

    best = study.best_params
    result = run_backtest(
        df, best["ma_period"], best["entry_distance"], best["protection_pct"]
    )

    print_optimization_summary(study)
    print_trade_statistics(result)
    generate_current_signal(df, best)


if __name__ == "__main__":
    main()
