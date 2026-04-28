"""Rolling VaR backtesting and coverage tests."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import chi2

from src.var_models import (
    historical_var,
    monte_carlo_var,
    parametric_gaussian_var,
    student_t_var,
)


VaRFunction = Callable[[pd.Series, float], float]

MODEL_FUNCTIONS: dict[str, VaRFunction] = {
    "Historical": historical_var,
    "Parametric Gaussian": parametric_gaussian_var,
    "Student-t": student_t_var,
    "Monte Carlo": monte_carlo_var,
}


def run_var_backtest(
    returns: pd.Series,
    model_name: str,
    confidence_level: float,
    window: int = 252,
) -> tuple[pd.DataFrame, dict[str, float | str]]:
    """Run a rolling-window VaR backtest for one model and confidence level."""
    if model_name not in MODEL_FUNCTIONS:
        raise ValueError(f"Unsupported model: {model_name}")
    clean_returns = returns.dropna()
    if len(clean_returns) <= window:
        raise ValueError("Returns series must be longer than the estimation window.")

    var_function = MODEL_FUNCTIONS[model_name]
    records = []

    for index in range(window, len(clean_returns)):
        training_window = clean_returns.iloc[index - window : index]
        realised_return = clean_returns.iloc[index]
        var_estimate = var_function(training_window, confidence_level)
        breach = realised_return < var_estimate
        records.append(
            {
                "date": clean_returns.index[index],
                "realised_return": realised_return,
                "var": var_estimate,
                "breach": breach,
            }
        )

    backtest = pd.DataFrame.from_records(records).set_index("date")
    metrics = calculate_backtest_metrics(backtest, confidence_level)
    metrics["model"] = model_name
    metrics["confidence_level"] = confidence_level
    return backtest, metrics


def calculate_backtest_metrics(
    backtest: pd.DataFrame,
    confidence_level: float,
) -> dict[str, float | str]:
    """Calculate VaR exception metrics and statistical coverage tests."""
    observations = len(backtest)
    alpha = 1 - confidence_level
    breaches = backtest["breach"].astype(bool)
    actual_breaches = int(breaches.sum())
    breach_rate = actual_breaches / observations if observations else math.nan
    breach_returns = backtest.loc[breaches, "realised_return"]

    kupiec_stat, kupiec_p_value = kupiec_test(actual_breaches, observations, alpha)
    christoffersen_stat, christoffersen_p_value = christoffersen_independence_test(
        breaches
    )

    return {
        "observations": observations,
        "expected_breaches": observations * alpha,
        "actual_breaches": actual_breaches,
        "breach_rate": breach_rate,
        "expected_breach_rate": alpha,
        "average_breach_loss": float(breach_returns.mean())
        if actual_breaches
        else math.nan,
        "maximum_breach_loss": float(breach_returns.min())
        if actual_breaches
        else math.nan,
        "kupiec_statistic": kupiec_stat,
        "kupiec_p_value": kupiec_p_value,
        "christoffersen_statistic": christoffersen_stat,
        "christoffersen_p_value": christoffersen_p_value,
        "traffic_light": basel_traffic_light(actual_breaches, observations, alpha),
    }


def kupiec_test(
    actual_breaches: int,
    observations: int,
    expected_breach_rate: float,
) -> tuple[float, float]:
    """Kupiec proportion-of-failures unconditional coverage test."""
    if observations == 0:
        return math.nan, math.nan

    observed_rate = actual_breaches / observations
    if actual_breaches == 0:
        likelihood_ratio = -2 * observations * math.log(1 - expected_breach_rate)
    elif actual_breaches == observations:
        likelihood_ratio = -2 * observations * math.log(expected_breach_rate)
    else:
        log_likelihood_expected = (
            (observations - actual_breaches) * math.log(1 - expected_breach_rate)
            + actual_breaches * math.log(expected_breach_rate)
        )
        log_likelihood_observed = (
            (observations - actual_breaches) * math.log(1 - observed_rate)
            + actual_breaches * math.log(observed_rate)
        )
        likelihood_ratio = -2 * (log_likelihood_expected - log_likelihood_observed)

    return float(likelihood_ratio), float(1 - chi2.cdf(likelihood_ratio, df=1))


def christoffersen_independence_test(
    breaches: pd.Series,
) -> tuple[float, float]:
    """Christoffersen independence test based on breach transition counts."""
    breach_values = breaches.astype(int).to_numpy()
    if len(breach_values) < 2:
        return math.nan, math.nan

    previous = breach_values[:-1]
    current = breach_values[1:]
    n00 = int(((previous == 0) & (current == 0)).sum())
    n01 = int(((previous == 0) & (current == 1)).sum())
    n10 = int(((previous == 1) & (current == 0)).sum())
    n11 = int(((previous == 1) & (current == 1)).sum())

    pi = _safe_ratio(n01 + n11, n00 + n01 + n10 + n11)
    pi01 = _safe_ratio(n01, n00 + n01)
    pi11 = _safe_ratio(n11, n10 + n11)

    restricted = _bernoulli_log_likelihood(n01 + n11, n00 + n10, pi)
    unrestricted = (
        _bernoulli_log_likelihood(n01, n00, pi01)
        + _bernoulli_log_likelihood(n11, n10, pi11)
    )
    likelihood_ratio = -2 * (restricted - unrestricted)

    if not np.isfinite(likelihood_ratio) or likelihood_ratio < 0:
        likelihood_ratio = 0.0

    return float(likelihood_ratio), float(1 - chi2.cdf(likelihood_ratio, df=1))


def basel_traffic_light(
    actual_breaches: int,
    observations: int,
    expected_breach_rate: float,
) -> str:
    """Classify exceptions using a simple Basel-style traffic light rule."""
    if observations <= 300 and math.isclose(expected_breach_rate, 0.01):
        if actual_breaches <= 4:
            return "Green"
        if actual_breaches <= 9:
            return "Yellow"
        return "Red"

    breach_rate = actual_breaches / observations
    if breach_rate <= 1.6 * expected_breach_rate:
        return "Green"
    if breach_rate <= 3.0 * expected_breach_rate:
        return "Yellow"
    return "Red"


def _bernoulli_log_likelihood(successes: int, failures: int, probability: float) -> float:
    if probability <= 0:
        return 0.0 if successes == 0 else -math.inf
    if probability >= 1:
        return 0.0 if failures == 0 else -math.inf
    return successes * math.log(probability) + failures * math.log(1 - probability)


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
