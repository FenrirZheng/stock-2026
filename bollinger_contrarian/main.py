from .backtest import run_backtest
from .config import (
    COMMISSION_RATE,
    END_DATE,
    N_TRIALS,
    START_DATE,
    STOP_LOSS_PCT,
    TICKER,
    TRAIN_RATIO,
)
from .data_fetcher import fetch_stock_data, split_train_test
from .optimizer import get_best_feasible_params, run_optimization
from .reporter import print_parameter_importance, print_report, print_trades


def main():
    print(f"下載 {TICKER} 歷史資料 ({START_DATE} ~ {END_DATE})...")
    df = fetch_stock_data(TICKER, START_DATE, END_DATE)
    print(f"取得 {len(df)} 筆交易日資料")

    train_df, test_df = split_train_test(df, TRAIN_RATIO)
    print(f"訓練集: {len(train_df)} 天, 測試集: {len(test_df)} 天")

    print(f"\n開始 Bayesian Optimization ({N_TRIALS} trials)...")
    study = run_optimization(train_df, N_TRIALS)

    # 取得滿足所有約束的最佳參數
    best_params = get_best_feasible_params(study, train_df)
    if best_params is None:
        print("未找到滿足所有約束的可行解。")
        return

    # 用最佳參數分別回測 in-sample 和 out-of-sample
    train_result = run_backtest(
        train_df, best_params["ma_period"], best_params["num_std"],
        STOP_LOSS_PCT, COMMISSION_RATE,
    )
    test_result = run_backtest(
        test_df, best_params["ma_period"], best_params["num_std"],
        STOP_LOSS_PCT, COMMISSION_RATE,
    )

    # 取得期間字串
    train_start = str(train_df.index[0].date())
    train_end = str(train_df.index[-1].date())
    test_start = str(test_df.index[0].date())
    test_end = str(test_df.index[-1].date())

    print_report(
        best_params, train_result, test_result,
        len(train_df), len(test_df),
        train_start, train_end, test_start, test_end,
    )
    print_trades(test_result.trades, "Out-of-sample ")
    print_parameter_importance(study)


if __name__ == "__main__":
    main()
