"""Portfolio construction utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily percentage returns from a price DataFrame."""
    returns = prices.pct_change().dropna()
    if returns.empty:
        raise ValueError("Not enough price observations to calculate returns.")
    return returns


def build_portfolio_returns(
    returns: pd.DataFrame,
    weights: np.ndarray | list[float] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Construct weighted portfolio returns from individual asset returns."""
    asset_count = len(returns.columns)
    portfolio_weights = (
        np.repeat(1 / asset_count, asset_count)
        if weights is None
        else np.asarray(weights, dtype=float)
    )

    if len(portfolio_weights) != asset_count:
        raise ValueError("Number of weights must match number of return columns.")
    if not np.isclose(portfolio_weights.sum(), 1.0):
        raise ValueError("Portfolio weights must sum to 1.")

    portfolio_returns = returns.dot(portfolio_weights)
    portfolio_returns.name = "portfolio_return"
    return returns, portfolio_returns
