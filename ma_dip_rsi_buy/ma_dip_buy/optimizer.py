from dataclasses import dataclass

import pandas as pd
from bayes_opt import BayesianOptimization

from .backtest import run_backtest
from .config import (
    INIT_POINTS,
    MAX_DRAWDOWN_LIMIT,
    MIN_TRADES,
    N_ITER,
    PBOUNDS,
    PENALTY_SCORE,
    RANDOM_STATE,
)


@dataclass
class OptimizeResult:
    best_params: dict
    best_score: float
    all_iterations: list[dict]


def create_objective(df: pd.DataFrame):
    """建立 objective 函數，DataFrame 透過 closure 傳入。"""

    def objective(x, m, n, k, t):
        x_int = int(round(x))
        n_int = int(round(n))

        # 排除無效組合：trailing stop 應比 hard stop 更敏感
        if t >= k:
            return PENALTY_SCORE

        result = run_backtest(df, x_int, m, n_int, k, t)

        if result.n_trades < MIN_TRADES:
            return PENALTY_SCORE

        if result.max_drawdown >= MAX_DRAWDOWN_LIMIT:
            return PENALTY_SCORE

        return result.score

    return objective


def run_optimization(
    df: pd.DataFrame,
    init_points: int = INIT_POINTS,
    n_iter: int = N_ITER,
    random_state: int = RANDOM_STATE,
    verbose: int = 0,
) -> OptimizeResult:
    """執行 Bayesian Optimization，回傳最佳參數與歷程。"""
    optimizer = BayesianOptimization(
        f=create_objective(df),
        pbounds=PBOUNDS,
        random_state=random_state,
        verbose=verbose,
    )
    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    best = optimizer.max
    best_params = best["params"].copy()
    best_params["x"] = int(round(best_params["x"]))
    best_params["n"] = int(round(best_params["n"]))

    all_iterations = [
        {"params": res["params"], "score": res["target"]}
        for res in optimizer.res
    ]

    return OptimizeResult(
        best_params=best_params,
        best_score=best["target"],
        all_iterations=all_iterations,
    )
