import pandas as pd

from .backtest import BacktestResult, compute_trade_return
from .config import TICKER
from .optimizer import OptimizeResult


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
    print(f"  MA Dip-Buy Strategy — {TICKER}")
    print("══════════════════════════════════════════════════")

    print()
    print("── 資料摘要 ──")
    print(f"  訓練期間: {train_start} ~ {train_end} ({train_size} 天)")
    print(f"  測試期間: {test_start} ~ {test_end} ({test_size} 天)")

    print()
    print("── 最佳參數 ──")
    print(f"  MA 週期 (x)           : {p['x']}")
    print(f"  買進觸發 (m)          : {p['m']:.2f}%")
    print(f"  未突破超時天數 (n)    : {p['n']}")
    print(f"  固定止損 (k)          : {p['k']:.2f}%")
    print(f"  Trailing Stop (t)     : {p['t']:.2f}%")

    print()
    print("── 訓練集績效 ──")
    _print_metrics(train_result)

    print()
    print("── 測試集績效（Out-of-Sample）──")
    _print_metrics(test_result)

    _print_overfitting_check(train_result, test_result)
    _print_exit_breakdown(train_result, "訓練集")
    _print_exit_breakdown(test_result, "測試集")

    if train_df is not None:
        _print_trade_details(train_result, train_df, "訓練集")
    if test_df is not None:
        _print_trade_details(test_result, test_df, "測試集")

    _print_convergence(opt_result)

    print()
    print("══════════════════════════════════════════════════")


def _print_metrics(result: BacktestResult) -> None:
    print(f"  交易筆數              : {result.n_trades}")
    print(f"  勝率                  : {result.win_rate:.2%}")
    print(f"  平均報酬              : {result.mean_return:.4%}")
    print(f"  Score (勝率×平均報酬) : {result.score:.6f}")
    print(f"  總報酬                : {result.total_return:.2%}")
    print(f"  最大回撤              : {result.max_drawdown:.2%}")


def _print_overfitting_check(
    train: BacktestResult, test: BacktestResult
) -> None:
    print()
    print("── 過擬合檢查 ──")
    score_gap = train.score - test.score
    wr_gap = train.win_rate - test.win_rate
    ret_gap = train.mean_return - test.mean_return
    print(f"  Score gap (Train-Test)     : {score_gap:+.6f}")
    print(f"  勝率 gap                   : {wr_gap:+.2%}")
    print(f"  平均報酬 gap               : {ret_gap:+.4%}")
    if test.score > 0 and score_gap / train.score < 0.3:
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


def _print_trade_details(
    result: BacktestResult, df: pd.DataFrame, label: str
) -> None:
    if result.n_trades == 0:
        return
    print()
    print(f"── 交易明細（{label}）──")
    print(f"  {'#':>3s}  {'進場日':>12s}  {'進場價':>9s}  "
          f"{'出場日':>12s}  {'出場價':>9s}  {'天數':>4s}  "
          f"{'報酬%':>8s}  {'出場原因'}")
    print(f"  {'─'*3}  {'─'*12}  {'─'*9}  "
          f"{'─'*12}  {'─'*9}  {'─'*4}  "
          f"{'─'*8}  {'─'*12}")
    for i, (tr, ret) in enumerate(
        zip(result.trades, result.returns), 1
    ):
        days_held = tr.exit_day - tr.entry_day
        print(f"  {i:>3d}  {tr.entry_date:>12s}  {tr.entry_price:>9.2f}  "
              f"{tr.exit_date:>12s}  {tr.exit_price:>9.2f}  {days_held:>4d}  "
              f"{ret:>+8.2%}  {tr.exit_reason}")


def _print_convergence(opt_result: OptimizeResult) -> None:
    print()
    print("── 優化收斂 ──")
    iterations = opt_result.all_iterations
    best_so_far = float("-inf")
    checkpoints = [5, 10, 20, 40, 60, 80, 100, 120]
    print(f"  {'Iter':>6s}    {'Best Score':>12s}")
    print(f"  {'─' * 6}    {'─' * 12}")
    for i, item in enumerate(iterations, 1):
        if item["score"] > best_so_far:
            best_so_far = item["score"]
        if i in checkpoints:
            print(f"  {i:>6d}    {best_so_far:>12.6f}")
    total = len(iterations)
    if total not in checkpoints:
        print(f"  {total:>6d}    {best_so_far:>12.6f}")
