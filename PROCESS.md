# Project Process: Deep Reinforcement Learning for Dynamic Stat Arb

This document outlines the step-by-step process followed in this project to develop a Reinforcement Learning (RL) based Statistical Arbitrage strategy. The project aims to replace static stat-arb pipelines with a dynamic learning-based system that adapts to regime shifts and learns adaptive trading rules.

## Motivation & Problem Statement

**Why Classical Stat Arb Fails:**
*   Correlations drift; cointegration breaks during regime shifts.
*   Distance / OU-based spreads impose linearity and stationarity.
*   Fixed z-score thresholds cannot adapt to market volatility.
*   Signals are noisy when pair selection is based only on historical correlation.

**Our Objective:**
To build a system that:
*   Finds economically meaningful pairs.
*   Learns adaptive trading rules.
*   Reacts to regime shifts without fixed thresholds.

---

## Step 1: Pairs Network Construction (`Step1_Pairs_Network.ipynb`)

**Objective:** To identify potential trading pairs and construct a stable, economically grounded graph of valid pairs.

**Key Activities:**
*   **Data Loading:** Loading historical price data, sector information, and fundamental ratios.
*   **Universe Selection:** Filtering for S&P 500 constituents and handling data validity intervals.
*   **Economic Network Construction:**
    *   Building a multi-layer graph based on industry & sub-industry co-membership.
    *   Incorporating supplier-customer and competitor edges (weighted).
*   **Pairs Identification:**
    *   Computing Spearman correlations within each economic cluster.
    *   Selecting the strongest non-duplicate pairs per cluster (top monthly in-sector pairs).
    *   **Why this matters:** Produces stable, economically grounded pairs instead of noisy correlation screens.

## Step 2: Feature Engineering (`Step2_TIC_Features.ipynb`)

**Objective:** To generate technical, fundamental, and spread-based features that serve as state observations for the RL agent.

**Key Activities:**
*   **Spread & Signal Engineering:**
    *   Calculating rolling hedge ratio ($\beta$), log-spread, z-score, and $\Delta$-spread.
    *   Computing volatility (rolling std) and half-life of mean reversion.
    *   Generating residual spreads via ETF factor regression (e.g., SPY, SIZE, MTUM) to create clean, de-factorized mean-reversion signals.
*   **Technical Factors:** Computing RSI, MACD, Bollinger Bands, and last 4 days' returns.
*   **Data Cleaning:** Handling missing values and ensuring data consistency across the trading universe.

## Step 3: RL Model Training (`Step3_RL_Model.ipynb`)

**Objective:** To train a Recurrent PPO (PPO-LSTM) agent to make trading decisions based on the generated features.

**Key Activities:**
*   **Environment Setup:** Defining a custom Gym environment (`gymnasium`) where:
    *   **State:** All engineered features + current position.
    *   **Actions:** Discrete space (Short / Flat / Long).
    *   **Reward:** $\Delta$PnL minus transaction costs.
*   **Model Selection:** Using Recurrent Proximal Policy Optimization (RecurrentPPO) from `sb3_contrib` to handle sequence structure and temporal dependencies.
*   **Training:** Training the agent on historical data to maximize risk-adjusted returns.

## Step 4: Backtesting (`Step4_Backtest.ipynb`)

**Objective:** To evaluate the performance of the trained RL agent on out-of-sample data using a dynamic trading setup.

**Key Activities:**
*   **Dynamic Trading Setup:**
    *   Monthly selection of top in-sector pairs.
    *   Allocating fixed capital per pair.
    *   Running daily inference with pair-specific Recurrent PPO models.
*   **Performance Analysis:** Aggregating PnL to portfolio equity and calculating metrics like Sharpe ratio and maximum drawdown.
*   **Visualization:** Plotting the equity curve and other relevant charts.

## Limitations & Possible Extensions

1.  **Static Pair Universe:** Currently static within a month; could be improved with a Rolling TIC Network to refine selection using rolling persistence.
2.  **Portfolio-level RL:** Shifting from per-pair RL to joint allocation decisions across spreads.
3.  **Microstructure Realism:** Modeling slippage, spread widening, and partial fills for better execution realism.
4.  **Algorithm Comparison:** Evaluating SAC/TD3 for continuous sizing and stability against Recurrent PPO.

## Dependencies

The project relies on several key Python libraries:
*   `numpy`, `pandas`: Data manipulation.
*   `scikit-learn`: Preprocessing and clustering.
*   `gymnasium`: RL environment standard.
*   `sb3_contrib`: Stable Baselines3 Contrib for advanced RL algorithms (RecurrentPPO).
*   `cvxpy`: Convex optimization.
*   `matplotlib`: Visualization.
