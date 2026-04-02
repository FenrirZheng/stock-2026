# ma_search/main.py

from .backtest import run_backtest
from .config import (
    BASELINE_MA,
    END_DATE,
    N_STARTUP_TRIALS,
    N_TOTAL_TRIALS,
    START_DATE,
    TICKER,
    TRAIN_RATIO,
)
from .data_fetcher import fetch_stock_data, split_train_test
from .optimizer import run_bayesian_search, run_brute_force_search
from .reporter import print_report


def main():
    print(f"下載 {TICKER} 歷史資料 ({START_DATE} ~ {END_DATE})...")
    df = fetch_stock_data(TICKER, START_DATE, END_DATE)
    print(f"取得 {len(df)} 筆交易日資料")

    train_df, test_df = split_train_test(df, TRAIN_RATIO)
    print(f"訓練集: {len(train_df)} 筆, 測試集: {len(test_df)} 筆")

    print(f"\n計算 Baseline MA({BASELINE_MA})...")
    baseline_train = run_backtest(train_df, BASELINE_MA)
    baseline_test = run_backtest(test_df, BASELINE_MA)

    print(f"執行 Bayesian Optimization ({N_TOTAL_TRIALS} trials)...")
    bo_result = run_bayesian_search(
        train_df, test_df,
        n_trials=N_TOTAL_TRIALS,
        n_startup=N_STARTUP_TRIALS,
    )

    print(f"執行 Brute-Force Search...")
    bf_result = run_brute_force_search(train_df, test_df)

    train_start = str(train_df.index[0].date())
    train_end = str(train_df.index[-1].date())
    test_start = str(test_df.index[0].date())
    test_end = str(test_df.index[-1].date())

    print_report(
        bo_result=bo_result,
        bf_result=bf_result,
        baseline_train=baseline_train,
        baseline_test=baseline_test,
        baseline_n=BASELINE_MA,
        train_size=len(train_df),
        test_size=len(test_df),
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
    )


if __name__ == "__main__":
    main()
