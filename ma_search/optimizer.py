# ma_search/optimizer.py

from dataclasses import dataclass, field
from typing import Callable

import optuna
import pandas as pd

from .backtest import run_backtest
from .config import MA_MAX, MA_MIN, N_STARTUP_TRIALS, N_TOTAL_TRIALS


@dataclass
class SearchResult:
    best_n: int
    best_train_sharpe: float
    best_test_sharpe: float
    all_results: dict[int, float] = field(default_factory=dict)
    trial_history: list[tuple[int, float]] = field(default_factory=list)


def create_objective(train_df: pd.DataFrame) -> Callable:
    """建立 Optuna objective 函數，DataFrame 透過 closure 傳入。"""

    def objective(trial: optuna.Trial) -> float:
        ma_period = trial.suggest_int("ma_period", MA_MIN, MA_MAX)
        result = run_backtest(train_df, ma_period)
        return result.sharpe_ratio

    return objective


def run_bayesian_search(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_trials: int = N_TOTAL_TRIALS,
    n_startup: int = N_STARTUP_TRIALS,
    seed: int = 42,
) -> SearchResult:
    """執行 Optuna TPE 優化，回傳搜尋結果。"""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(create_objective(train_df), n_trials=n_trials)

    best_n = study.best_params["ma_period"]
    best_train_sharpe = study.best_value
    best_test_sharpe = run_backtest(test_df, best_n).sharpe_ratio

    trial_history: list[tuple[int, float]] = []
    best_so_far = float("-inf")
    for t in study.trials:
        if t.value is not None and t.value > best_so_far:
            best_so_far = t.value
        trial_history.append((t.number + 1, best_so_far))

    return SearchResult(
        best_n=best_n,
        best_train_sharpe=best_train_sharpe,
        best_test_sharpe=best_test_sharpe,
        all_results={},
        trial_history=trial_history,
    )


def run_brute_force_search(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> SearchResult:
    """暴力搜尋所有 n ∈ [MA_MIN, MA_MAX]，回傳搜尋結果。"""
    all_results: dict[int, float] = {}

    for n in range(MA_MIN, MA_MAX + 1):
        result = run_backtest(train_df, n)
        all_results[n] = result.sharpe_ratio

    best_n = max(all_results, key=all_results.get)
    best_train_sharpe = all_results[best_n]
    best_test_sharpe = run_backtest(test_df, best_n).sharpe_ratio

    trial_history: list[tuple[int, float]] = []
    best_so_far = float("-inf")
    for i, n in enumerate(range(MA_MIN, MA_MAX + 1), 1):
        if all_results[n] > best_so_far:
            best_so_far = all_results[n]
        trial_history.append((i, best_so_far))

    return SearchResult(
        best_n=best_n,
        best_train_sharpe=best_train_sharpe,
        best_test_sharpe=best_test_sharpe,
        all_results=all_results,
        trial_history=trial_history,
    )
