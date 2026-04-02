import optuna
import pandas as pd

from .backtest import BacktestResult, run_backtest
from .data_fetcher import add_moving_average


def print_optimization_summary(study: optuna.Study) -> None:
    """印出 BO 優化結果摘要。"""
    best = study.best_params
    importance = optuna.importance.get_param_importances(study)

    print()
    print("══════════════════════════════════════════════════")
    print("  Bayesian Optimization 結果")
    print("══════════════════════════════════════════════════")
    print(f"  最佳參數：")
    print(f"    MA 週期 (x₁)      : {best['ma_period']} 天")
    print(f"    進場距離 (x₂)      : {best['entry_distance']:.2f}%")
    print(f"    保護百分比 (x₃)    : {best['protection_pct']:.2f}%")
    print(f"  最大累積報酬率       : {study.best_value:+.2%}")
    print()
    print("  參數重要性 (fANOVA)：")

    name_map = {
        "ma_period": "x₁ MA 週期",
        "entry_distance": "x₂ 進場距離",
        "protection_pct": "x₃ 保護百分比",
    }

    for param, imp in importance.items():
        label = name_map.get(param, param)
        bar_len = int(imp * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"    {label:16s}: {bar}  {imp:.0%}")

    print("══════════════════════════════════════════════════")


def print_trade_statistics(result: BacktestResult) -> None:
    """印出交易統計。"""
    print()
    print("── 交易統計 ──")

    if result.trade_count == 0:
        print("  無交易紀錄")
        return

    wins = sum(1 for t in result.trades if t.return_pct > 0)
    losses = result.trade_count - wins

    print(f"  交易次數: {result.trade_count}")
    print(f"  勝率    : {result.win_rate:.1%}  ({wins} 勝 / {losses} 負)")
    print(f"  平均獲利: {result.avg_win:+.2%}")
    print(f"  平均虧損: {result.avg_loss:+.2%}")

    if result.avg_loss != 0:
        win_loss_ratio = abs(result.avg_win / result.avg_loss)
        print(f"  盈虧比  : {win_loss_ratio:.2f}")

    print(f"  累積報酬: {result.cumulative_return:+.2%}")
    print()
    print("  出場類型分佈：")
    exit_labels = {
        "protection": "保護出場",
        "ma_cross": "均線交叉出場",
        "end_of_data": "回測結束平倉",
    }
    for exit_type, count in result.exit_type_counts.items():
        label = exit_labels.get(exit_type, exit_type)
        pct = count / result.trade_count
        print(f"    {label:14s}: {count:3d} ({pct:.1%})")


def generate_current_signal(df: pd.DataFrame, best_params: dict) -> str:
    """根據最佳參數判斷當前市場訊號。"""
    ma_period = best_params["ma_period"]
    entry_distance = best_params["entry_distance"]
    protection_pct = best_params["protection_pct"]

    df_ma = add_moving_average(df, ma_period)
    last = df_ma.iloc[-1]
    close = last["Close"]
    ma = last["MA"]
    last_date = df_ma.index[-1]
    date_str = str(last_date.date()) if hasattr(last_date, "date") else str(last_date)

    distance_pct = (close - ma) / ma * 100
    entry_threshold = ma * (1 - entry_distance / 100)

    # 檢查是否在持倉中（回溯最近的進出場狀態）
    result = run_backtest(df, ma_period, entry_distance, protection_pct)

    if result.trades:
        last_trade = result.trades[-1]
        if last_trade.exit_type == "end_of_data":
            # 回測結束仍持有 → 目前持倉中
            protection_threshold = last_trade.buy_price * (1 + protection_pct / 100)
            signal = "持有中"
            reason = (
                f"持倉成本 {last_trade.buy_price:.1f}，"
                f"保護線 {protection_threshold:.1f}"
            )
        else:
            # 最近一筆已平倉 → 等待進場
            if close < entry_threshold:
                signal = "買進"
                reason = f"收盤 {close:.1f} 低於進場門檻 {entry_threshold:.1f}"
            else:
                signal = "觀望"
                reason = (
                    f"距離 MA 為 {distance_pct:+.2f}%，"
                    f"尚未達到進場門檻 -{entry_distance:.2f}%"
                )
    else:
        signal = "觀望"
        reason = "無歷史交易紀錄，條件未觸發"

    print()
    print("── 當前市場訊號 ──")
    print(f"  日期      : {date_str}")
    print(f"  收盤      : {close:.1f}")
    print(f"  MA({ma_period:d})   : {ma:.1f}")
    print(f"  距離 MA   : {distance_pct:+.2f}%")
    print(f"  進場門檻  : {entry_threshold:.1f} (MA × {1 - entry_distance/100:.4f})")
    print(f"  訊號      : {signal}")
    print(f"  原因      : {reason}")

    return signal
