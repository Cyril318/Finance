# Deep Hedging Project

This project implements a Deep Hedging agent using a Recurrent Neural Network (LSTM/GRU) to hedge a European Call Option under transaction costs. It follows the methodology inspired by "Deep Hedging" (Hans Buehler et al.).

## Project Structure

```
Deep_Hedging/
│
├── market_simulator.py    # Market data generator (GBM & Heston)
├── models.py              # Deep Learning model (LSTM)
├── losses.py              # Custom loss function (Exponential Utility with Transaction Costs)
├── trainer.py             # Training loop and data preparation
├── main.py                # Main orchestration script (Phases 4-6)
├── utils.py               # Black-Scholes formulas for comparison
└── __init__.py
```

## Setup

1.  **Install Dependencies:**
    Ensure you have Python 3.8+ installed. Install the required packages:
    ```bash
    pip install numpy pandas matplotlib scipy tensorflow
    ```

## Usage

To run the full pipeline (training, evaluation, and plotting):

```bash
# Run from the project root
export PYTHONPATH=$PYTHONPATH:.
python Deep_Hedging/main.py
```

## Methodology

### Phase 1: Market Simulator
Generates synthetic market data using Geometric Brownian Motion (GBM) for training and Heston Model for robustness testing.

### Phase 2: Deep Hedging Model
A Keras model using LSTM layers to process the sequence of market features (Log Returns, Moneyness, Time to Maturity) and output the hedging strategy (Delta).

### Phase 3: Loss Function
Minimizes the expected exponential utility of the final wealth:
$$ \mathcal{L} = \mathbb{E} [ -\exp(-\lambda \cdot \text{Wealth}_T) ] $$
Where Wealth includes the option payoff and transaction costs:
$$ \text{Wealth}_T = - \text{Payoff}(S_T) + \sum_{t=0}^{T-1} (\delta_t \Delta S_t - \text{TransactionCosts}_t) $$

### Phases 4-6: Experiments
-   **Phase 4 (Sanity Check):** Training with 0 transaction costs. The model learns to replicate Black-Scholes Delta.
-   **Phase 5 (Frictions):** Training with transaction costs (e.g., 20 bps). The model learns a "No-Action Band" strategy to reduce turnover.
-   **Phase 6 (Analysis):** Compares PnL distributions and tests robustness on Heston data.

## Results

After running the script, check the `results/` directory for plots:
-   `phase4_sanity_check.png`: Comparison of AI Delta vs Black-Scholes Delta (should match closely).
-   `phase5_frictions.png`: Comparison under costs (AI should trade less frequently).
-   `phase6_pnl_dist.png`: PnL distribution comparison.
-   `phase6_heston_test.png`: Performance on Heston data (Model Risk).
