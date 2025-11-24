# src/dataloader.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional
import pandas as pd
import yfinance as yf

def fetch_prices(
    tickers: Iterable[str],
    start: str,
    end: str,
    interval: str = "1d",
    auto_adjust: bool = True,
    cache_path: Path | None = Path("data/prices.csv"),

) -> pd.DataFrame:
    """
    Download historical prices for tickers.
    Returns a DataFrame with columns per ticker (Adj Close).
    """
    tickers = list(tickers)
    if cache_path and cache_path.exists():
        return pd.read_csv(cache_path, parse_dates=["Date"], index_col="Date")

    data = yf.download(
        tickers=" ".join(tickers),
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
    )
    
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])
    elif isinstance(data.columns, pd.MultiIndex):
        cols = data.columns.get_level_values(0)
        price_field = "Adj Close" if "Adj Close" in cols else "Close"
        data = data[price_field]
    else:
        if "Adj Close" in data.columns:
            data = data[["Adj Close"]]
            data.columns = tickers
        elif "Close" in data.columns and len(tickers) == 1:
            data = data[["Close"]]
            data.columns = tickers
    data = data.dropna(how="all")
    data.index.name = "Date"

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(cache_path)
    return data

