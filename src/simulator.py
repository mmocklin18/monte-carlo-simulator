from pathlib import Path
from dataloader import fetch_prices

prices = fetch_prices(
    tickers=["SPY", "AGG"],
    start="2013-01-01",
    end="2023-12-31"

)
print(prices.head())
daily_returns = prices.pct_change().dropna()
print(daily_returns)
