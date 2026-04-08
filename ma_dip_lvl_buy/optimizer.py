from dataclasses import dataclass

import optuna
import pandas as pd

from .backtest import run_backtest
from .config import (
    MAX_DRAWDOWN_LIMIT,
    MIN_TRADES,
    N_STARTUP_TRIALS,
    N_TRIALS,
    PARAM_RANGES,
    RANDOM_STATE,
)


@dataclass
class OptimizeResult:
    best_params: dict
    best_score: float
    all_trials: list[dict]
    param_importance: dict[str, float] | None


def create_objective(df: pd.DataFrame):
    """建立 Optuna objective。以 Sharpe Ratio 為優化目標。"""

    def objective(trial: optuna.Trial) -> float:
        ma_period = trial.suggest_int(
            "ma_period", *PARAM_RANGES["ma_period"])
        dip_pct = trial.suggest_float(
            "dip_pct", *PARAM_RANGES["dip_pct"])
        rsi_threshold = trial.suggest_float(
            "rsi_threshold", *PARAM_RANGES["rsi_threshold"])
        timeout_days = trial.suggest_int(
            "timeout_days", *PARAM_RANGES["timeout_days"])
        hard_stop_pct = trial.suggest_float(
            "hard_stop_pct", *PARAM_RANGES["hard_stop_pct"])
        trail_stop_pct = trial.suggest_float(
            "trail_stop_pct", *PARAM_RANGES["trail_stop_pct"])

        # trailing stop 應比 hard stop 更敏感
        if trail_stop_pct >= hard_stop_pct:
            raise optuna.TrialPruned()

        # dip_pct 不應超過 hard_stop_pct，否則進場時已接近止損價位
        if dip_pct >= hard_stop_pct:
            raise optuna.TrialPruned()

        result = run_backtest(df, ma_period, dip_pct, rsi_threshold,
                              timeout_days, hard_stop_pct, trail_stop_pct)

        if result.n_trades < MIN_TRADES:
            raise optuna.TrialPruned()

        if result.max_drawdown >= MAX_DRAWDOWN_LIMIT:
            raise optuna.TrialPruned()

        return result.sharpe_ratio

    return objective


def run_optimization(
    df: pd.DataFrame,
    n_trials: int = N_TRIALS,
    random_state: int = RANDOM_STATE,
    verbose: int = 0,
) -> OptimizeResult:
    """執行 Optuna TPE 優化，回傳最佳參數與歷程。"""
    sampler = optuna.samplers.TPESampler(
        seed=random_state,
        n_startup_trials=N_STARTUP_TRIALS,
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
    )

    verbosity = optuna.logging.WARNING if verbose == 0 else optuna.logging.INFO
    optuna.logging.set_verbosity(verbosity)

    study.optimize(create_objective(df), n_trials=n_trials)

    best = study.best_trial
    best_params = best.params.copy()

    all_trials = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            all_trials.append({
                "params": t.params,
                "score": t.value,
            })

    # 參數重要度分析（fANOVA）
    param_importance = None
    try:
        importance = optuna.importance.get_param_importances(study)
        param_importance = dict(importance)
    except Exception:
        pass

    return OptimizeResult(
        best_params=best_params,
        best_score=best.value,
        all_trials=all_trials,
        param_importance=param_importance,
    )
