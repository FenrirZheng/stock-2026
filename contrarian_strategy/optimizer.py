from typing import Callable

import optuna
import pandas as pd

from .backtest import run_backtest
from .config import ENTRY_DISTANCE_RANGE, MA_RANGE, N_TRIALS, PROTECTION_RANGE


def create_objective(df: pd.DataFrame) -> Callable:
    """建立 Optuna objective 函數，DataFrame 透過 closure 傳入。"""

    def objective(trial: optuna.Trial) -> float:
        ma_period = trial.suggest_int("ma_period", *MA_RANGE)
        entry_distance = trial.suggest_float("entry_distance", *ENTRY_DISTANCE_RANGE)
        protection_pct = trial.suggest_float("protection_pct", *PROTECTION_RANGE)

        result = run_backtest(df, ma_period, entry_distance, protection_pct)
        return result.cumulative_return

    return objective


def run_optimization(
    df: pd.DataFrame, n_trials: int = N_TRIALS, seed: int = 42
) -> optuna.Study:
    """執行 Bayesian Optimization，回傳 Optuna Study。"""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(create_objective(df), n_trials=n_trials)

    return study


def get_parameter_importance(study: optuna.Study) -> dict[str, float]:
    """用 fANOVA 計算各參數的重要性。"""
    return optuna.importance.get_param_importances(study)
