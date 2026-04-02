from typing import Callable

import optuna
import pandas as pd

from .backtest import run_backtest
from .config import (
    COMMISSION_RATE,
    LAMBDA_PENALTY,
    MA_RANGE,
    MAX_DRAWDOWN,
    MIN_TRADES,
    N_STARTUP_TRIALS,
    N_TRIALS,
    STD_RANGE,
    STOP_LOSS_PCT,
)


def create_objective(train_df: pd.DataFrame) -> Callable:
    """建立 Optuna objective 函數，DataFrame 透過 closure 傳入。"""

    def objective(trial: optuna.Trial) -> float:
        ma_period = trial.suggest_int("ma_period", *MA_RANGE)
        num_std = trial.suggest_float("num_std", *STD_RANGE)

        result = run_backtest(
            train_df, ma_period, num_std, STOP_LOSS_PCT, COMMISSION_RATE
        )

        # 硬約束：交易次數不足或回撤過大 → prune
        if result.trade_count < MIN_TRADES:
            raise optuna.TrialPruned()
        if result.max_drawdown > MAX_DRAWDOWN:
            raise optuna.TrialPruned()

        # WinRate - λ·max(0, -TotalReturn)
        penalty = LAMBDA_PENALTY * max(0, -result.total_return)
        return result.win_rate - penalty

    return objective


def run_optimization(
    train_df: pd.DataFrame,
    n_trials: int = N_TRIALS,
    seed: int = 42,
) -> optuna.Study:
    """執行 Bayesian Optimization，回傳 Optuna Study。"""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = optuna.samplers.TPESampler(
        seed=seed, n_startup_trials=N_STARTUP_TRIALS
    )
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(create_objective(train_df), n_trials=n_trials)

    return study


def get_best_feasible_params(
    study: optuna.Study,
    train_df: pd.DataFrame,
) -> dict | None:
    """從 study 中找出滿足所有約束的最佳參數。

    若 best_trial 的 TotalReturn < 0，往下搜尋可行解。
    """
    sorted_trials = sorted(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
        key=lambda t: t.value,
        reverse=True,
    )

    for trial in sorted_trials:
        params = trial.params
        result = run_backtest(
            train_df,
            params["ma_period"],
            params["num_std"],
            STOP_LOSS_PCT,
            COMMISSION_RATE,
        )
        if result.total_return >= 0:
            return params

    return None
