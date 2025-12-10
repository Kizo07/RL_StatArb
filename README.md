# RL StatArb Final Project

**Course:** UCLA MFE Applied AI  
**Project:** Reinforcement Learning for Statistical Arbitrage

This repository contains the code and resources for the final project on using Reinforcement Learning (RL) for Statistical Arbitrage. The project implements a complete pipeline from data processing and pairs selection to model training and backtesting.

## Project Structure

The project is organized into four sequential steps, each implemented in a Jupyter Notebook:

1.  **[Step1_Pairs_Network.ipynb](Step1_Pairs_Network.ipynb)**:  
    Constructs the pairs network by identifying relationships between assets using sector information and clustering techniques.

2.  **[Step2_TIC_Features.ipynb](Step2_TIC_Features.ipynb)**:  
    Generates technical and fundamental features (TIC features) used as inputs for the RL model.

3.  **[Step3_RL_Model.ipynb](Step3_RL_Model.ipynb)**:  
    Trains a Recurrent PPO (Proximal Policy Optimization) agent using `sb3_contrib` and a custom Gymnasium environment.

4.  **[Step4_Backtest.ipynb](Step4_Backtest.ipynb)**:  
    Backtests the trained model on out-of-sample data to evaluate its performance.

## Detailed Process

For a detailed explanation of the methodology and steps involved, please refer to **[PROCESS.md](PROCESS.md)**.

## Getting Started

### Prerequisites

The project requires Python and the following major libraries:
*   `numpy`
*   `pandas`
*   `scikit-learn`
*   `gymnasium`
*   `sb3_contrib`
*   `cvxpy`
*   `matplotlib`

You can install the necessary packages using pip:

```bash
pip install numpy pandas scikit-learn gymnasium sb3-contrib cvxpy matplotlib
```

### Running the Code

The notebooks were originally designed to run in a Google Colab environment.
*   **Data Paths:** You may need to adjust the file paths (currently pointing to `/content/drive/My Drive/...`) to match your local directory structure or mount your Google Drive if running on Colab.
*   **Execution Order:** It is recommended to run the notebooks in sequential order (Step 1 to Step 4) as later steps depend on artifacts (like `.feather` or `.pkl` files) generated in previous steps.

## Presentation

A summary presentation of the project is available here: [Advanced AI Presentation final.pdf](Advanced%20AI%20Presentation%20final.pdf)

## License

See the [LICENSE](LICENSE) file for details.
