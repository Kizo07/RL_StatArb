# Project Process: RL Statistical Arbitrage

This document outlines the step-by-step process followed in this project to develop a Reinforcement Learning (RL) based Statistical Arbitrage strategy. The project is divided into four main stages, each corresponding to a Jupyter Notebook.

## Step 1: Pairs Network Construction (`Step1_Pairs_Network.ipynb`)

**Objective:** To identify potential trading pairs and construct a network of related assets.

**Key Activities:**
*   **Data Loading:** Loading historical price data, sector information, and fundamental ratios.
*   **Universe Selection:** Filtering for S&P 500 constituents and handling data validity intervals.
*   **Clustering & Grouping:** Using sector information (GICS sub-industries) to group similar stocks.
*   **Pairs Identification:** Analyzing relationships between stocks to identify potential pairs for statistical arbitrage. This involves pearson correlation analysis on the residuals, finding overlapping validity periods for economically linked stocks.

## Step 2: Feature Engineering (`Step2_TIC_Features.ipynb`)

**Objective:** To generate technical and fundamental features that will serve as state observations for the RL agent.

**Key Activities:**
*   **Data Preparation:** Consolidating price and ratio data.
*   **ETF Analysis:** Incorporating data from factor ETFs (e.g., SPY, SIZE, MTUM, VLUE, QUAL) to capture market conditions.
*   **Feature Calculation:** Computing technical indicators (TIC features), fundamental signals and statarb signals.
*   **Data Cleaning:** Handling missing values and ensuring data consistency across the trading universe.

## Step 3: RL Model Training (`Step3_RL_Model.ipynb`)

**Objective:** To train a Reinforcement Learning agent to make trading decisions based on the generated features.

**Key Activities:**
*   **Environment Setup:** Defining a custom Gym environment (`gymnasium`) that simulates the trading market. This environment handles state transitions, action execution, and reward calculation.
*   **Model Selection:** Using the Recurrent Proximal Policy Optimization (RecurrentPPO) algorithm from `sb3_contrib`. Recurrent models are chosen to capture temporal dependencies in financial time series.
*   **Training:** Training the agent on historical data to maximize the reward function (likely risk-adjusted returns).
*   **Feature Loading:** Loading the pre-computed features (`tech_features.feather`, etc.) to feed into the model.

## Step 4: Backtesting (`Step4_Backtest.ipynb`)

**Objective:** To evaluate the performance of the trained RL agent on out-of-sample data.

**Key Activities:**
*   **Model Loading:** Loading the trained RecurrentPPO model.
*   **Simulation:** Running the agent through the test period using the same environment setup.
*   **Performance Analysis:** Calculating key performance metrics such as cumulative returns, Sharpe ratio, and maximum drawdown.
*   **Visualization:** Plotting the equity curve and other relevant charts to visualize the strategy's performance.

## Dependencies

The project relies on several key Python libraries:
*   `numpy`, `pandas`: Data manipulation.
*   `scikit-learn`: Preprocessing and clustering.
*   `gymnasium`: RL environment standard.
*   `sb3_contrib`: Stable Baselines3 Contrib for advanced RL algorithms (RecurrentPPO).
*   `cvxpy`: Convex optimization (likely for portfolio optimization or constraints).
*   `matplotlib`: Visualization.
