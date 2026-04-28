"""Plotting helpers for the VaR backtesting engine."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_returns_distribution(
    returns: pd.Series,
    historical_var_99: float,
    parametric_var_99: float,
    output_path: Path,
) -> None:
    """Save a histogram of portfolio returns with 99% VaR thresholds."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(returns, bins=60, color="#9bb7d4", edgecolor="white", alpha=0.85)
    ax.axvline(
        historical_var_99,
        color="#c0392b",
        linestyle="--",
        linewidth=2,
        label=f"Historical VaR 99% ({historical_var_99:.2%})",
    )
    ax.axvline(
        parametric_var_99,
        color="#2c3e50",
        linestyle="--",
        linewidth=2,
        label=f"Parametric VaR 99% ({parametric_var_99:.2%})",
    )
    ax.set_title("Daily Portfolio Returns Distribution")
    ax.set_xlabel("Daily return")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_rolling_var_99(
    rolling_results: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    """Save realised returns and rolling 99% VaR estimates by model."""
    fig, ax = plt.subplots(figsize=(12, 7))
    first_result = next(iter(rolling_results.values()))
    ax.plot(
        first_result.index,
        first_result["realised_return"],
        color="#606f7b",
        linewidth=0.8,
        label="Realised return",
    )

    colors = {
        "Historical": "#c0392b",
        "Parametric Gaussian": "#2c3e50",
        "Monte Carlo": "#1f8a70",
    }
    for model_name, result in rolling_results.items():
        ax.plot(
            result.index,
            result["var"],
            linewidth=1.2,
            label=f"{model_name} VaR 99%",
            color=colors.get(model_name),
        )

    breach_result = rolling_results["Historical"]
    breaches = breach_result[breach_result["breach"]]
    ax.scatter(
        breaches.index,
        breaches["realised_return"],
        color="#e67e22",
        s=22,
        label="Historical VaR breaches",
        zorder=3,
    )

    ax.set_title("Rolling 99% VaR Backtest")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily return")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_var_model_comparison(
    static_results: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save a grouped bar chart comparing VaR and ES across models."""
    labels = [
        f"{row.model}\n{int(row.confidence_level * 100)}% {row.metric}"
        for row in static_results.itertuples()
    ]
    values = static_results["value"]
    colors = ["#4c78a8" if metric == "VaR" else "#f58518" for metric in static_results["metric"]]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(labels, values, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("VaR and Expected Shortfall by Model")
    ax.set_ylabel("Daily return threshold")
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_cumulative_returns(
    portfolio_returns: pd.Series,
    output_path: Path,
) -> None:
    """Save cumulative return of the portfolio."""
    cumulative_returns = (1 + portfolio_returns).cumprod() - 1
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(cumulative_returns.index, cumulative_returns, color="#1f8a70", linewidth=1.8)
    ax.set_title("50/50 AAPL-MSFT Portfolio Cumulative Return")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
