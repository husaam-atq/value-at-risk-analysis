"""Run the Portfolio Value-at-Risk Backtesting Engine."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.backtesting import MODEL_FUNCTIONS, run_var_backtest
from src.data import DEFAULT_END_DATE, DEFAULT_START_DATE, DEFAULT_TICKERS, load_price_data
from src.plots import (
    plot_cumulative_returns,
    plot_returns_distribution,
    plot_rolling_var_99,
    plot_var_model_comparison,
)
from src.portfolio import build_portfolio_returns, calculate_returns
from src.var_models import calculate_var_es_by_model


CONFIDENCE_LEVELS = (0.95, 0.99)
OUTPUT_DIR = Path("outputs")


def main() -> None:
    """Run the full data, risk estimation, backtesting, and reporting pipeline."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    prices = load_price_data(
        tickers=DEFAULT_TICKERS,
        start_date=DEFAULT_START_DATE,
        end_date=DEFAULT_END_DATE,
    )
    individual_returns = calculate_returns(prices)
    _, portfolio_returns = build_portfolio_returns(individual_returns)

    static_results = _build_static_results(portfolio_returns)
    static_results.to_csv(OUTPUT_DIR / "static_var_results.csv", index=False)

    backtest_results, rolling_var_99 = _run_backtests(portfolio_returns)
    backtest_results.to_csv(OUTPUT_DIR / "backtest_results.csv", index=False)

    historical_var_99 = _lookup_static_value(
        static_results, "Historical", 0.99, "VaR"
    )
    parametric_var_99 = _lookup_static_value(
        static_results, "Parametric Gaussian", 0.99, "VaR"
    )

    plot_returns_distribution(
        portfolio_returns,
        historical_var_99,
        parametric_var_99,
        OUTPUT_DIR / "returns_distribution.png",
    )
    plot_rolling_var_99(rolling_var_99, OUTPUT_DIR / "rolling_var_99.png")
    plot_var_model_comparison(static_results, OUTPUT_DIR / "var_model_comparison.png")
    plot_cumulative_returns(portfolio_returns, OUTPUT_DIR / "cumulative_returns.png")

    _print_summary(static_results, backtest_results)


def _build_static_results(portfolio_returns: pd.Series) -> pd.DataFrame:
    records = []
    for confidence_level in CONFIDENCE_LEVELS:
        model_results = calculate_var_es_by_model(portfolio_returns, confidence_level)
        for model_name, metrics in model_results.items():
            for metric_name, value in metrics.items():
                records.append(
                    {
                        "model": model_name,
                        "confidence_level": confidence_level,
                        "metric": metric_name,
                        "value": value,
                    }
                )
    return pd.DataFrame.from_records(records)


def _run_backtests(
    portfolio_returns: pd.Series,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    metrics_records = []
    rolling_var_99 = {}

    for confidence_level in CONFIDENCE_LEVELS:
        for model_name in MODEL_FUNCTIONS:
            backtest, metrics = run_var_backtest(
                portfolio_returns,
                model_name=model_name,
                confidence_level=confidence_level,
                window=252,
            )
            metrics_records.append(metrics)
            if confidence_level == 0.99:
                rolling_var_99[model_name] = backtest

    return pd.DataFrame.from_records(metrics_records), rolling_var_99


def _lookup_static_value(
    static_results: pd.DataFrame,
    model: str,
    confidence_level: float,
    metric: str,
) -> float:
    row = static_results[
        (static_results["model"] == model)
        & (static_results["confidence_level"] == confidence_level)
        & (static_results["metric"] == metric)
    ]
    return float(row.iloc[0]["value"])


def _print_summary(
    static_results: pd.DataFrame,
    backtest_results: pd.DataFrame,
) -> None:
    print("\nPortfolio Value-at-Risk Backtesting Engine")
    print("================================================")
    print("Static VaR / ES")
    print(_format_static_results(static_results).to_string(index=False))
    print("\nRolling Backtest Results")
    print(_format_backtest_results(backtest_results).to_string(index=False))
    print("\nOutputs saved to the outputs/ directory.")


def _format_static_results(static_results: pd.DataFrame) -> pd.DataFrame:
    formatted = static_results.copy()
    formatted["confidence_level"] = formatted["confidence_level"].map("{:.0%}".format)
    formatted["value"] = formatted["value"].map("{:.2%}".format)
    return formatted


def _format_backtest_results(backtest_results: pd.DataFrame) -> pd.DataFrame:
    formatted = backtest_results.copy()
    percentage_columns = [
        "breach_rate",
        "expected_breach_rate",
        "average_breach_loss",
        "maximum_breach_loss",
    ]
    for column in percentage_columns:
        formatted[column] = formatted[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.2%}"
        )
    for column in [
        "expected_breaches",
        "kupiec_statistic",
        "kupiec_p_value",
        "christoffersen_statistic",
        "christoffersen_p_value",
    ]:
        formatted[column] = formatted[column].map("{:.4f}".format)
    formatted["confidence_level"] = formatted["confidence_level"].map("{:.0%}".format)
    return formatted[
        [
            "model",
            "confidence_level",
            "observations",
            "expected_breaches",
            "actual_breaches",
            "breach_rate",
            "kupiec_p_value",
            "christoffersen_p_value",
            "traffic_light",
        ]
    ]


if __name__ == "__main__":
    main()
