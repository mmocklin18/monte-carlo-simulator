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

## To Run:
First run command "mamba activate py3XD" to activate environment

Next choose desired parameters in main.py

To run, ensure you are in the "code" directory and use command: "python3 main.py"
optional flags:
    --scenario: "bull", "bear", "volatile" (market conditions)
    --paths: int (number of path simulations)
    --rebalance: int (# of steps until rebalance)


