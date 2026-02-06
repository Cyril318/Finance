# CLAUDE.md - AI Assistant Guide for Deep Hedging Project

## Project Overview

This is a **Deep Hedging** research project implementing neural network-based option hedging using LSTM/GRU models in TensorFlow/Keras. It follows the methodology from "Deep Hedging" (Hans Buehler et al.) to hedge a European Call Option under transaction costs.

The core idea: train a recurrent neural network to learn an optimal hedging strategy that minimizes exponential utility loss, accounting for transaction costs — outperforming classical Black-Scholes delta hedging in the presence of market frictions.

## Repository Structure

```
Finance/
├── CLAUDE.md                      # This file
├── README.md                      # Project documentation
└── Deep_Hedging/                  # Main Python package
    ├── __init__.py                # Package init (empty)
    ├── main.py                    # Orchestration: phases 4-6 experiments
    ├── market_simulator.py        # MarketSimulator: GBM and Heston models
    ├── models.py                  # create_deep_hedging_model(): LSTM architecture
    ├── losses.py                  # DeepHedgingLoss: exponential utility loss
    ├── trainer.py                 # train_hedging_model() + prepare_data()
    └── utils.py                   # Black-Scholes pricing and delta formulas
```

**Runtime output:** `results/` directory (created by `main.py`) containing `.png` plots.

## Tech Stack

- **Language:** Python 3.8+
- **Deep Learning:** TensorFlow / Keras (LSTM, GRU layers)
- **Numerical:** NumPy, SciPy (scipy.stats.norm)
- **Visualization:** Matplotlib
- **No formal test framework, CI/CD, or linting tools configured**

## Setup and Running

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install numpy pandas matplotlib scipy tensorflow

# Run the full pipeline (from project root)
export PYTHONPATH=$PYTHONPATH:.
python Deep_Hedging/main.py
```

There is no `requirements.txt` — dependencies are listed in README.md.

## Module Guide

### `market_simulator.py` — MarketSimulator class
- `simulate_gbm(n_simulations, mu, sigma)` — Geometric Brownian Motion paths
- `simulate_heston(n_simulations, mu, kappa, theta, xi, rho, v0=None)` — Heston stochastic volatility paths
- All outputs shape: `(n_simulations, n_steps + 1, 1)`

### `models.py` — Neural Network
- `create_deep_hedging_model(n_steps, n_features, units=32, activation='sigmoid')` — Returns a Keras `Model`
- Architecture: `Input → LSTM(units, return_sequences=True) → TimeDistributed(Dense(1))`
- Output is delta (hedge ratio) at each timestep, constrained to [0, 1] via sigmoid

### `losses.py` — DeepHedgingLoss class
- Custom `tf.keras.losses.Loss` subclass
- Computes: `E[exp(-λ * Wealth)]` where `Wealth = PnL - TransactionCosts - Payoff`
- Parameters: `risk_aversion` (λ), `transaction_cost` (rate), `strike`
- Falls back to `-E[Wealth]` when `risk_aversion ≈ 0`

### `trainer.py` — Training Pipeline
- `prepare_data(prices, strike, T, dt)` — Constructs 3 features per timestep: log return, log moneyness, normalized TTM
- `train_hedging_model(...)` — Full pipeline: simulate → prepare → build model → compile → fit
- Default: Adam optimizer, lr=0.001, batch_size=256, 20% validation split

### `utils.py` — Black-Scholes Reference
- `bs_call_price(S, K, T, r, sigma)` — Closed-form European call price
- `bs_call_delta(S, K, T, r, sigma)` — Closed-form delta (N(d1))
- Handles T→0 edge cases

### `main.py` — Experiment Runner
- **Phase 4 (Sanity Check):** Train with 0 transaction costs, verify AI delta ≈ BS delta
- **Phase 5 (Frictions):** Train with 20bps costs, observe "No-Action Band" behavior
- **Phase 6 (Analysis):** Compare PnL distributions; test robustness on Heston-generated data
- Helper functions: `evaluate_model()`, `calculate_pnl()`

## Key Conventions

### Data Shapes
- Prices: `(n_simulations, n_steps + 1, 1)` — includes initial price
- Deltas/Features: `(n_simulations, n_steps, n_features)` — one per decision point
- The +1 in prices accounts for the terminal price needed for payoff calculation

### Parameters
- Default option: ATM European Call, S0=K=100, T=30/365 (30 days), σ=0.2, μ=0
- Training: 50,000 simulations (increase to 1M for publication-quality results), 5 epochs
- Transaction cost rate: 0.002 (20 basis points)

### Import Pattern
The package uses relative imports internally (e.g., `from .models import ...`). External code imports via `from Deep_Hedging.trainer import ...`, which requires the project root on `PYTHONPATH`.

## Development Notes

- No tests exist — validate changes by running `main.py` and inspecting output plots
- No linter/formatter configured — code follows standard Python conventions
- No `.gitignore` — be careful not to commit `results/`, `__pycache__/`, `.venv/`, or large model files
- The `results/` directory is created at runtime via `os.makedirs("results", exist_ok=True)`
- TensorFlow warnings are expected on first run (GPU/CUDA detection)

## Common Modifications

- **Change model architecture:** Edit `models.py` — swap LSTM for GRU, adjust `units`
- **Add features:** Edit `prepare_data()` in `trainer.py` — update `n_features` in `main.py` model creation
- **Adjust loss function:** Edit `DeepHedgingLoss.call()` in `losses.py`
- **Add new stochastic models:** Add methods to `MarketSimulator` in `market_simulator.py`
- **Change training params:** Modify arguments in `train_hedging_model()` calls in `main.py`
