import optuna

from .backtest import BacktestResult
from .config import COMMISSION_RATE, STOP_LOSS_PCT, TICKER


def print_report(
    best_params: dict,
    train_result: BacktestResult,
    test_result: BacktestResult,
    train_size: int,
    test_size: int,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
) -> None:
    """印出完整回測報告，含 in-sample / out-of-sample 對照。"""
    print()
    print("══════════════════════════════════════════════════")
    print(f"  Bollinger Channel Strategy — {TICKER}")
    print("══════════════════════════════════════════════════")

    print()
    print("── 資料摘要 ──")
    print(f"  訓練期間: {train_start} ~ {train_end} ({train_size} 天)")
    print(f"  測試期間: {test_start} ~ {test_end} ({test_size} 天)")
    print(f"  手續費  : 單邊 {COMMISSION_RATE:.1%}")
    print(f"  停損    : {STOP_LOSS_PCT:.0%} (固定)")

    print()
    print("── 最佳參數 ──")
    print(f"  SMA 週期 (x)       : {best_params['ma_period']}")
    print(f"  標準差倍數 (y)     : {best_params['num_std']:.2f}")

    print()
    _print_comparison(train_result, test_result)
    _print_overfitting(train_result, test_result)

    print()
    print("══════════════════════════════════════════════════")


def _print_comparison(
    train: BacktestResult, test: BacktestResult
) -> None:
    """印出 In-sample vs Out-of-sample 對照表。"""
    print("── In-sample vs Out-of-sample ──")
    print(f"  {'指標':14s} {'In-sample':>12s}  {'Out-of-sample':>14s}")
    print(f"  {'─' * 14} {'─' * 12}  {'─' * 14}")
    print(f"  {'勝率':14s} {train.win_rate:>11.2%}  {test.win_rate:>13.2%}")
    print(f"  {'總報酬':14s} {train.total_return:>+11.2%}  {test.total_return:>+13.2%}")
    print(f"  {'最大回撤':14s} {-train.max_drawdown:>11.2%}  {-test.max_drawdown:>13.2%}")
    print(f"  {'交易次數':14s} {train.trade_count:>11d}  {test.trade_count:>13d}")
    print(f"  {'Profit Factor':14s} {train.profit_factor:>11.2f}  {test.profit_factor:>13.2f}")


def _print_overfitting(
    train: BacktestResult, test: BacktestResult
) -> None:
    """印出過擬合檢查。"""
    print()
    print("── 過擬合檢查 ──")
    wr_gap = train.win_rate - test.win_rate
    ret_gap = train.total_return - test.total_return
    print(f"  WinRate gap      : {wr_gap:+.2%}  "
          f"(Train {train.win_rate:.2%} → Test {test.win_rate:.2%})")
    print(f"  TotalReturn gap  : {ret_gap:+.2%}  "
          f"(Train {train.total_return:+.2%} → Test {test.total_return:+.2%})")


def print_trades(trades: list, label: str = "") -> None:
    """印出每筆交易明細。"""
    if not trades:
        print(f"\n── 交易明細 {label} ── (無交易)")
        return

    print(f"\n── 交易明細 {label}({len(trades)} 筆) ──")
    print(f"  {'#':>3s}  {'買進日':10s}  {'買價':>8s}  {'賣出日':10s}  {'賣價':>8s}  {'報酬':>8s}  {'出場'}")
    print(f"  {'─'*3}  {'─'*10}  {'─'*8}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*10}")
    for i, t in enumerate(trades, 1):
        print(
            f"  {i:3d}  {t.buy_date:10s}  {t.buy_price:8.1f}  "
            f"{t.sell_date:10s}  {t.sell_price:8.1f}  "
            f"{t.pnl:+7.2%}  {t.exit_type}"
        )


def print_parameter_importance(study: optuna.Study) -> None:
    """印出參數重要性。"""
    try:
        importance = optuna.importance.get_param_importances(study)
    except Exception:
        print("\n  參數重要性：資料不足，無法計算")
        return

    name_map = {
        "ma_period": "x  SMA 週期",
        "num_std": "y  標準差倍數",
    }

    print()
    print("── 參數重要性 (fANOVA) ──")
    for param, imp in importance.items():
        label = name_map.get(param, param)
        bar_len = int(imp * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {label:16s}: {bar}  {imp:.0%}")
