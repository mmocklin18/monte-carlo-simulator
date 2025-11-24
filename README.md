# Monte Carlo Portfolio Simulator

Simulate multi-asset portfolios with GBM, correlations, rebalancing, and scenario analysis (bull/bear/volatile). Includes yfinance download and cached prices.


## Structure
```bash
src/dataloader.py   # fetch/cached prices
src/gbm.py          # GBM path generation
src/portfolio.py    # weights, rebalancing
src/scenarios.py    # scenario testing
src/visualize.py    # plots
src/simulator.py    # orchestrates run
data/prices.csv     # cached sample data
```
