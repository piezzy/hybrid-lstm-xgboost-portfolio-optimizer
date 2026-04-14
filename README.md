# Hybrid LSTM-XGBoost Portfolio Optimizer

A hybrid deep learning and machine learning framework for stock return prediction and portfolio optimization in the Indonesian equity market, enhanced with explainable artificial intelligence (XAI).

---

## Overview

This project integrates **LSTM** (temporal modeling) and **XGBoost** (non-linear modeling) using a dynamic ensemble weighted by validation performance.  
Backtesting is conducted using a walk-forward approach over the 2020-2025 period on LQ45 stocks.

---

## Backtest Results (2020–2025)

| Metric            | Strategy | IHSG    |
| ----------------- | -------- | ------- |
| Total Return      | 325.18%  | 35.41%  |
| Annualized Return | 27.75%   | 5.26%   |
| Sharpe Ratio      | 1.08     | 0.03    |
| Max Drawdown      | -26.90%  | -37.75% |
| Alpha             | 24.89%   | -       |
| Beta              | 0.43     | -       |

---

## Key Features

- 21-day return prediction using LSTM–XGBoost ensemble
- Walk-forward backtesting (22 rebalancing periods)
- Portfolio optimization via Sharpe ratio maximization (SLSQP)
- Volatility-based dynamic trailing stop-loss (5%–12%)
- Sector diversification (max 30% per sector, 20% per stock)
- Explainable AI: XGBoost feature importance and LSTM saliency
- Transaction cost: 0.2% buy and 0.2% sell

---

## Project Structure

```
├── data-extraction.ipynb
├── hybrid-portfolio-optimizer.ipynb
├── all_data.csv
├── IHSG.csv
└── backtest_results.csv
```

---

## Requirements

```bash
pip install tensorflow xgboost scikit-learn pandas numpy
pip install statsmodels scipy matplotlib seaborn tabulate
```

---

## Usage

1. Run `data-extraction.ipynb` to prepare the dataset
2. Run `hybrid-portfolio-optimizer.ipynb` sequentially
3. Results are saved to `backtest_results.csv`

---

## Model Architecture

**LSTM**  
Input (84, 18) → LSTM(64) → BatchNormalization → LSTM(32) → Dense(16) → Dropout(0.3) → Dense(1)

**XGBoost**  
n_estimators = 300, max_depth = 4, learning_rate = 0.02, subsample = 0.8

**Ensemble**  
Dynamic weighting based on inverse validation MSE

---

## Features

- **Momentum:** 5d, 10d, 21d, 63d
- **Volatility:** volatility_21d, vol_adj_mom
- **Technical:** RSI, normalized RSI, MA(20/50), trend strength
- **Price action:** drawdown, daily range, body ratio, price vs high
- **Return statistics:** mean and standard deviation (21d log returns)
- **Volume:** volume ratio and log ratio

---

## Disclaimer

This project is for academic research purposes only.  
Past performance does not guarantee future results.

---

## Author

Mohammad Hikam 'Abdul Karim - Telkom University Purwokerto - 2026
