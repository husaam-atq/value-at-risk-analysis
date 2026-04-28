"""Market data loading utilities."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import yfinance as yf


DEFAULT_TICKERS = ["AAPL", "MSFT"]
DEFAULT_START_DATE = "2018-01-01"
DEFAULT_END_DATE = "2023-12-31"


def load_price_data(
    tickers: list[str] | None = None,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance.

    yfinance's ``end`` argument is exclusive, so one calendar day is added to
    make the public function's end date intuitive for users.
    """
    selected_tickers = tickers or DEFAULT_TICKERS
    inclusive_end = pd.to_datetime(end_date) + timedelta(days=1)

    raw = yf.download(
        selected_tickers,
        start=start_date,
        end=inclusive_end.strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
    )

    if raw.empty:
        raise ValueError("No price data was downloaded. Check tickers and date range.")

    prices = _extract_adjusted_close(raw, selected_tickers)
    prices = prices.dropna().sort_index()

    if prices.empty:
        raise ValueError("Price data is empty after dropping missing values.")

    return prices


def _extract_adjusted_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Return adjusted close prices, falling back to close if needed."""
    if isinstance(raw.columns, pd.MultiIndex):
        field_level = 0 if "Adj Close" in raw.columns.get_level_values(0) else None
        if field_level is not None:
            prices = raw["Adj Close"]
        elif "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"]
        else:
            raise ValueError("Downloaded data does not contain close prices.")
    else:
        if "Adj Close" in raw.columns:
            prices = raw[["Adj Close"]].copy()
        elif "Close" in raw.columns:
            prices = raw[["Close"]].copy()
        else:
            raise ValueError("Downloaded data does not contain close prices.")
        prices.columns = tickers[:1]

    return prices.loc[:, [ticker for ticker in tickers if ticker in prices.columns]]
