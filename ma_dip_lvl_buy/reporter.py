import pandas as pd

from .backtest import BacktestResult
from .config import TICKER
from .optimizer import OptimizeResult


def compute_buy_and_hold(df: pd.DataFrame) -> float:
    """計算同期 Buy & Hold 報酬率。"""
    if len(df) < 2:
        return 0.0
    first_close = float(df["Close"].iloc[0])
    last_close = float(df["Close"].iloc[-1])
    return (last_close - first_close) / first_close


def print_report(
    opt_result: OptimizeResult,
    train_result: BacktestResult,
    test_result: BacktestResult,
    train_size: int,
    test_size: int,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    train_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
) -> None:
    """印出完整優化結果報告。"""
    p = opt_result.best_params

    print()
    print("══════════════════════════════════════════════════")
    print(f"  MA Dip-Buy LVL Strategy — {TICKER}")
    print("══════════════════════════════════════════════════")

    print()
    print("── 資料摘要 ──")
    print(f"  訓練期間: {train_start} ~ {train_end} ({train_size} 天)")
    print(f"  測試期間: {test_start} ~ {test_end} ({test_size} 天)")

    print()
    print("── 最佳參數 ──")
    print(f"  MA 週期              : {p['ma_period']}")
    print(f"  買進觸發 (dip_pct)   : {p['dip_pct']:.2f}%")
    print(f"  RSI 門檻             : {p['rsi_threshold']:.1f}")
    print(f"  未突破超時天數       : {p['timeout_days']}")
    print(f"  固定止損 (hard_stop) : {p['hard_stop_pct']:.2f}%")
    print(f"  Trailing Stop        : {p['trail_stop_pct']:.2f}%")

    print()
    print("── 訓練集績效 ──")
    _print_metrics(train_result)

    print()
    print("── 測試集績效（Out-of-Sample）──")
    _print_metrics(test_result)

    # Buy & Hold 對比
    if train_df is not None and test_df is not None:
        _print_benchmark(train_result, test_result, train_df, test_df)

    _print_overfitting_check(train_result, test_result)
    _print_exit_breakdown(train_result, "訓練集")
    _print_exit_breakdown(test_result, "測試集")

    if train_df is not None:
        _print_trade_details(train_result, "訓練集")
    if test_df is not None:
        _print_trade_details(test_result, "測試集")

    if opt_result.param_importance:
        _print_param_importance(opt_result.param_importance)

    _print_convergence(opt_result)

    print()
    print("══════════════════════════════════════════════════")


def _print_metrics(result: BacktestResult) -> None:
    print(f"  交易筆數              : {result.n_trades}")
    print(f"  勝率                  : {result.win_rate:.2%}")
    print(f"  平均報酬              : {result.mean_return:.4%}")
    print(f"  總報酬                : {result.total_return:.2%}")
    print(f"  年化報酬              : {result.annualized_return:.2%}")
    print(f"  年化波動度            : {result.annualized_volatility:.2%}")
    print(f"  Sharpe Ratio          : {result.sharpe_ratio:.4f}")
    print(f"  Profit Factor         : {result.profit_factor:.2f}")
    print(f"  最大回撤（日頻）      : {result.max_drawdown:.2%}")
    print(f"  Exposure Ratio        : {result.exposure_ratio:.2%}")


def _print_benchmark(
    train_result: BacktestResult,
    test_result: BacktestResult,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    train_bh = compute_buy_and_hold(train_df)
    test_bh = compute_buy_and_hold(test_df)

    print()
    print("── Buy & Hold 對比 ──")
    print(f"  {'':20s}  {'策略':>10s}  {'B&H':>10s}  {'超額':>10s}")
    print(f"  {'訓練集':20s}  {train_result.total_return:>+10.2%}"
          f"  {train_bh:>+10.2%}"
          f"  {train_result.total_return - train_bh:>+10.2%}")
    print(f"  {'測試集':20s}  {test_result.total_return:>+10.2%}"
          f"  {test_bh:>+10.2%}"
          f"  {test_result.total_return - test_bh:>+10.2%}")


def _print_overfitting_check(
    train: BacktestResult, test: BacktestResult
) -> None:
    print()
    print("── 過擬合檢查 ──")
    sharpe_gap = train.sharpe_ratio - test.sharpe_ratio
    wr_gap = train.win_rate - test.win_rate
    ret_gap = train.mean_return - test.mean_return
    print(f"  Sharpe gap (Train-Test)    : {sharpe_gap:+.4f}")
    print(f"  勝率 gap                   : {wr_gap:+.2%}")
    print(f"  平均報酬 gap               : {ret_gap:+.4%}")
    if test.sharpe_ratio > 0 and (
        train.sharpe_ratio == 0
        or sharpe_gap / train.sharpe_ratio < 0.3
    ):
        print("  → 測試集表現尚可，過擬合風險較低")
    else:
        print("  → ⚠ 測試集衰退明顯，可能過擬合")


def _print_exit_breakdown(result: BacktestResult, label: str) -> None:
    if result.n_trades == 0:
        return
    print()
    print(f"── 出場原因分布（{label}）──")
    reasons: dict[str, int] = {}
    for tr in result.trades:
        reasons[tr.exit_reason] = reasons.get(tr.exit_reason, 0) + 1
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        pct = count / result.n_trades
        print(f"  {reason:<20s}: {count:>4d} ({pct:.1%})")


def _print_trade_details(result: BacktestResult, label: str) -> None:
    if result.n_trades == 0:
        return
    print()
    print(f"── 交易明細（{label}）──")
    print(f"  {'#':>3s}  {'訊號日':>12s}  {'進場日':>12s}  {'進場價':>9s}  "
          f"{'出場日':>12s}  {'出場價':>9s}  {'天數':>4s}  "
          f"{'報酬%':>8s}  {'出場原因'}")
    print(f"  {'─'*3}  {'─'*12}  {'─'*12}  {'─'*9}  "
          f"{'─'*12}  {'─'*9}  {'─'*4}  "
          f"{'─'*8}  {'─'*12}")
    for i, (tr, ret) in enumerate(
        zip(result.trades, result.returns), 1
    ):
        print(f"  {i:>3d}  {tr.signal_date:>12s}  {tr.entry_date:>12s}"
              f"  {tr.entry_price:>9.2f}  "
              f"{tr.exit_date:>12s}  {tr.exit_price:>9.2f}  {tr.days_held:>4d}  "
              f"{ret:>+8.2%}  {tr.exit_reason}")


def _print_param_importance(importance: dict[str, float]) -> None:
    print()
    print("── 參數重要度（fANOVA）──")
    for param, value in sorted(importance.items(), key=lambda x: -x[1]):
        bar = "█" * int(value * 40)
        print(f"  {param:<20s}: {value:.3f}  {bar}")


def _print_convergence(opt_result: OptimizeResult) -> None:
    print()
    print("── 優化收斂 ──")
    trials = opt_result.all_trials
    if not trials:
        print("  （無完成的試驗）")
        return

    best_so_far = float("-inf")
    checkpoints = [5, 10, 20, 40, 60, 80, 100, 150, 200]
    print(f"  {'Trial':>6s}    {'Best Sharpe':>12s}")
    print(f"  {'─' * 6}    {'─' * 12}")
    for i, item in enumerate(trials, 1):
        if item["score"] > best_so_far:
            best_so_far = item["score"]
        if i in checkpoints:
            print(f"  {i:>6d}    {best_so_far:>12.4f}")
    total = len(trials)
    if total not in checkpoints:
        print(f"  {total:>6d}    {best_so_far:>12.4f}")
