from .backtest import run_backtest
from .config import (
    END_DATE,
    N_TRIALS,
    START_DATE,
    TICKER,
    TRAIN_RATIO,
)
from .data_fetcher import fetch_stock_data, split_train_test
from .optimizer import run_optimization
from .reporter import print_report


def main():
    print(f"下載 {TICKER} 歷史資料 ({START_DATE} ~ {END_DATE})...")
    df = fetch_stock_data(TICKER, START_DATE, END_DATE)
    print(f"取得 {len(df)} 筆交易日資料")

    train_df, test_df = split_train_test(df, TRAIN_RATIO)
    print(f"訓練集: {len(train_df)} 筆, 測試集: {len(test_df)} 筆")

    print(f"\n執行 Optuna TPE Optimization (trials={N_TRIALS})...")
    opt_result = run_optimization(train_df)

    p = opt_result.best_params
    print(f"最佳參數: ma={p['ma_period']}, dip={p['dip_pct']:.2f}%, "
          f"rsi<{p['rsi_threshold']:.1f}, timeout={p['timeout_days']}, "
          f"hard_stop={p['hard_stop_pct']:.2f}%, "
          f"trail={p['trail_stop_pct']:.2f}%")

    print("\n以最佳參數回測訓練集...")
    train_result = run_backtest(
        train_df, p["ma_period"], p["dip_pct"], p["rsi_threshold"],
        p["timeout_days"], p["hard_stop_pct"], p["trail_stop_pct"])

    print("以最佳參數回測測試集...")
    test_result = run_backtest(
        test_df, p["ma_period"], p["dip_pct"], p["rsi_threshold"],
        p["timeout_days"], p["hard_stop_pct"], p["trail_stop_pct"])

    train_start = str(train_df.index[0].date())
    train_end = str(train_df.index[-1].date())
    test_start = str(test_df.index[0].date())
    test_end = str(test_df.index[-1].date())

    print_report(
        opt_result=opt_result,
        train_result=train_result,
        test_result=test_result,
        train_size=len(train_df),
        test_size=len(test_df),
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        train_df=train_df,
        test_df=test_df,
    )


if __name__ == "__main__":
    main()
