"""Value-at-Risk and Expected Shortfall models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


SUPPORTED_CONFIDENCE_LEVELS = (0.95, 0.99)


def historical_var(returns: pd.Series, confidence_level: float) -> float:
    """Estimate Historical VaR as the lower-tail return quantile."""
    _validate_confidence_level(confidence_level)
    return float(np.quantile(_clean_returns(returns), 1 - confidence_level))


def historical_es(returns: pd.Series, confidence_level: float) -> float:
    """Estimate Historical Expected Shortfall below the VaR threshold."""
    var = historical_var(returns, confidence_level)
    tail_returns = _clean_returns(returns)[_clean_returns(returns) <= var]
    return float(tail_returns.mean()) if len(tail_returns) else var


def parametric_gaussian_var(returns: pd.Series, confidence_level: float) -> float:
    """Estimate Gaussian VaR from the empirical mean and standard deviation."""
    _validate_confidence_level(confidence_level)
    clean = _clean_returns(returns)
    mean = clean.mean()
    std = clean.std(ddof=1)
    return float(mean + std * norm.ppf(1 - confidence_level))


def parametric_gaussian_es(returns: pd.Series, confidence_level: float) -> float:
    """Estimate Gaussian Expected Shortfall from empirical moments."""
    _validate_confidence_level(confidence_level)
    clean = _clean_returns(returns)
    alpha = 1 - confidence_level
    z_alpha = norm.ppf(alpha)
    mean = clean.mean()
    std = clean.std(ddof=1)
    return float(mean - std * norm.pdf(z_alpha) / alpha)


def monte_carlo_var(
    returns: pd.Series,
    confidence_level: float,
    simulations: int = 20_000,
    seed: int = 42,
) -> float:
    """Estimate Monte Carlo VaR using a Gaussian simulation calibrated to returns."""
    simulated_returns = _simulate_returns(returns, simulations, seed)
    return float(np.quantile(simulated_returns, 1 - confidence_level))


def monte_carlo_es(
    returns: pd.Series,
    confidence_level: float,
    simulations: int = 20_000,
    seed: int = 42,
) -> float:
    """Estimate Monte Carlo Expected Shortfall using calibrated simulations."""
    simulated_returns = _simulate_returns(returns, simulations, seed)
    var = np.quantile(simulated_returns, 1 - confidence_level)
    tail_returns = simulated_returns[simulated_returns <= var]
    return float(tail_returns.mean()) if len(tail_returns) else float(var)


def calculate_var_es_by_model(
    returns: pd.Series,
    confidence_level: float,
) -> dict[str, dict[str, float]]:
    """Calculate VaR and ES for each supported model."""
    return {
        "Historical": {
            "VaR": historical_var(returns, confidence_level),
            "ES": historical_es(returns, confidence_level),
        },
        "Parametric Gaussian": {
            "VaR": parametric_gaussian_var(returns, confidence_level),
            "ES": parametric_gaussian_es(returns, confidence_level),
        },
        "Monte Carlo": {
            "VaR": monte_carlo_var(returns, confidence_level),
            "ES": monte_carlo_es(returns, confidence_level),
        },
    }


def _simulate_returns(
    returns: pd.Series,
    simulations: int,
    seed: int,
) -> np.ndarray:
    clean = _clean_returns(returns)
    rng = np.random.default_rng(seed)
    return rng.normal(clean.mean(), clean.std(ddof=1), simulations)


def _clean_returns(returns: pd.Series) -> pd.Series:
    clean = returns.dropna()
    if clean.empty:
        raise ValueError("Returns series is empty.")
    return clean


def _validate_confidence_level(confidence_level: float) -> None:
    if confidence_level not in SUPPORTED_CONFIDENCE_LEVELS:
        supported = ", ".join(f"{level:.0%}" for level in SUPPORTED_CONFIDENCE_LEVELS)
        raise ValueError(f"Unsupported confidence level. Use one of: {supported}.")
