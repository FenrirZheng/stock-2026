# max_sharpe_ma/reporter.py

from .backtest import BacktestResult
from .config import TICKER
from .optimizer import SearchResult


def print_report(
    bo_result: SearchResult,
    bf_result: SearchResult,
    baseline_train: BacktestResult,
    baseline_test: BacktestResult,
    baseline_n: int,
    train_size: int,
    test_size: int,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
) -> None:
    """印出完整比較報告。"""
    print()
    print("══════════════════════════════════════════════════")
    print(f"  MA Crossover Period Search — {TICKER}")
    print("══════════════════════════════════════════════════")

    print()
    print("── 資料摘要 ──")
    print(f"  訓練期間: {train_start} ~ {train_end} ({train_size} 天)")
    print(f"  測試期間: {test_start} ~ {test_end} ({test_size} 天)")
    print(f"  搜尋空間: n ∈ [{bo_result.trial_history[0][0]}, ...], "
          f"共 {len(bf_result.all_results)} 個候選值")

    print()
    print(f"── Bayesian Optimization ({len(bo_result.trial_history)} trials, "
          f"含 random startup) ──")
    print(f"  最佳 MA 週期 (n) : {bo_result.best_n}")
    print(f"  Train Sharpe     : {bo_result.best_train_sharpe:.4f}")
    print(f"  Test  Sharpe     : {bo_result.best_test_sharpe:.4f}")

    print()
    print(f"── Brute-Force Search ({len(bf_result.all_results)} evaluations) ──")
    print(f"  最佳 MA 週期 (n) : {bf_result.best_n}")
    print(f"  Train Sharpe     : {bf_result.best_train_sharpe:.4f}")
    print(f"  Test  Sharpe     : {bf_result.best_test_sharpe:.4f}")

    print()
    print(f"── Baseline: MA({baseline_n}) ──")
    print(f"  Train Sharpe     : {baseline_train.sharpe_ratio:.4f}")
    print(f"  Test  Sharpe     : {baseline_test.sharpe_ratio:.4f}")

    _print_convergence(bo_result, bf_result)
    _print_overfitting_check(bo_result, bf_result)

    print()
    print("══════════════════════════════════════════════════")


def _print_convergence(bo_result: SearchResult, bf_result: SearchResult) -> None:
    """印出收斂比較表。"""
    print()
    print("── 收斂比較 ──")
    print(f"  {'Evals':>7s}    {'BO':>8s}    {'Brute-Force':>12s}")
    print(f"  {'─' * 7}    {'─' * 8}    {'─' * 12}")

    checkpoints = [5, 10, 15, 20]
    for cp in checkpoints:
        bo_val = _get_best_at(bo_result.trial_history, cp)
        bf_val = _get_best_at(bf_result.trial_history, cp)
        bo_str = f"{bo_val:.4f}" if bo_val is not None else "   -   "
        bf_str = f"{bf_val:.4f}" if bf_val is not None else "     -     "
        print(f"  {cp:>7d}    {bo_str:>8s}    {bf_str:>12s}")


def _get_best_at(
    history: list[tuple[int, float]], eval_count: int
) -> float | None:
    """取得第 eval_count 次評估時的最佳 Sharpe。"""
    for idx, val in history:
        if idx == eval_count:
            return val
    return None


def _print_overfitting_check(bo_result: SearchResult, bf_result: SearchResult) -> None:
    """印出過擬合檢查。"""
    print()
    print("── 過擬合檢查 ──")
    bo_gap = bo_result.best_train_sharpe - bo_result.best_test_sharpe
    bf_gap = bf_result.best_train_sharpe - bf_result.best_test_sharpe
    print(f"  BO : Train→Test gap: {bo_gap:+.4f}  "
          f"(Train {bo_result.best_train_sharpe:.4f} → Test {bo_result.best_test_sharpe:.4f})")
    print(f"  BF : Train→Test gap: {bf_gap:+.4f}  "
          f"(Train {bf_result.best_train_sharpe:.4f} → Test {bf_result.best_test_sharpe:.4f})")
