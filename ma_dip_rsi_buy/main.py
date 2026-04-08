from .backtest import run_backtest
from .config import (
    END_DATE,
    INIT_POINTS,
    N_ITER,
    SPLIT_DATE,
    START_DATE,
    TICKER,
)
from .data_fetcher import fetch_stock_data, split_train_test
from .optimizer import run_optimization
from .reporter import print_report


def main():
    print(f"下載 {TICKER} 歷史資料 ({START_DATE} ~ {END_DATE})...")
    df = fetch_stock_data(TICKER, START_DATE, END_DATE)
    print(f"取得 {len(df)} 筆交易日資料")

    train_df, test_df = split_train_test(df, SPLIT_DATE)
    print(f"訓練集: {len(train_df)} 筆, 測試集: {len(test_df)} 筆")

    print(f"\n執行 Bayesian Optimization "
          f"(init={INIT_POINTS}, iter={N_ITER})...")
    opt_result = run_optimization(train_df)

    p = opt_result.best_params
    print(f"最佳參數: x={p['x']}, m={p['m']:.2f}, n={p['n']}, "
          f"k={p['k']:.2f}, t={p['t']:.2f}, "
          f"rsi_threshold={p['rsi_threshold']:.2f}")

    print("\n以最佳參數回測訓練集...")
    train_result = run_backtest(
        train_df, p["x"], p["m"], p["n"], p["k"], p["t"],
        p["rsi_threshold"])

    print("以最佳參數回測測試集...")
    test_result = run_backtest(
        test_df, p["x"], p["m"], p["n"], p["k"], p["t"],
        p["rsi_threshold"])

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
