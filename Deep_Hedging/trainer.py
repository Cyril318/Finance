import numpy as np
import tensorflow as tf
from .market_simulator import MarketSimulator
from .models import create_deep_hedging_model
from .losses import DeepHedgingLoss

def prepare_data(prices, strike, T, dt):
    """
    Prepares input features for the deep hedging model.

    Args:
        prices: (n_simulations, n_steps + 1, 1)
        strike: Strike price
        T: Time to maturity
        dt: Time step

    Returns:
        inputs: (n_simulations, n_steps, n_features)
        prices: (n_simulations, n_steps + 1, 1) passed through as targets
    """
    # Prices S_t for t=0 to T-1
    S_t = prices[:, :-1, :]
    # Prices S_{t-1} needed for log returns.
    # For t=0, we don't have S_{-1}.
    # Log return at t=0: ln(S_0 / S_{-1}).
    # We can assume S_{-1} = S_0, so log_ret = 0.

    # Or simply:
    # Log Returns ln(S_t / S_{t-1})
    # For t=0, use 0.
    # For t > 0, ln(S_t / S_{t-1})

    # Let's compute log returns for the whole path and shift.
    # prices: S_0, S_1, ..., S_T
    # log_prices = ln(prices)
    # diff = ln(S_{i}) - ln(S_{i-1})

    log_prices = np.log(prices)
    log_returns_full = np.diff(log_prices, axis=1) # (n_sims, n_steps, 1)
    # This gives log returns for intervals [0,1], [1,2], ...
    # But at step t (decision time), we know S_t and S_{t-1}.
    # So the feature at t=0 should be "past return", maybe 0.
    # The feature at t=1 should be ln(S_1 / S_0).

    # So we pad a 0 at the beginning and take first n_steps.
    zeros = np.zeros((prices.shape[0], 1, 1))
    log_returns_input = np.concatenate([zeros, log_returns_full[:, :-1, :]], axis=1)

    # Log Moneyness: ln(S_t / K)
    log_moneyness = np.log(S_t / strike)

    # Time to Maturity: T - t * dt
    n_steps = S_t.shape[1]
    time_indices = np.arange(n_steps) # 0, 1, ..., N-1
    times = time_indices * dt
    time_to_maturity = T - times
    # Normalize TTM
    time_to_maturity_norm = time_to_maturity / T

    # Broadcast TTM to (n_sims, n_steps, 1)
    ttm_feature = np.tile(time_to_maturity_norm[np.newaxis, :, np.newaxis], (prices.shape[0], 1, 1))

    # Concatenate features
    # features: [log_return, log_moneyness, time_to_maturity]
    features = np.concatenate([log_returns_input, log_moneyness, ttm_feature], axis=2)

    return features, prices

def train_hedging_model(
    n_simulations=100000,
    n_steps=30,
    S0=100,
    strike=100,
    T=30/365,
    mu=0.0,
    sigma=0.2,
    risk_aversion=1.0,
    transaction_cost=0.0,
    batch_size=256,
    epochs=10,
    validation_split=0.2
):
    """
    Trains the Deep Hedging model.
    """
    dt = T / n_steps

    # 1. Simulate Market Data (GBM)
    # We use risk-neutral drift (mu=0 or r) for pricing/hedging training usually.
    # The prompt says "mu" input. We'll use provided mu.
    simulator = MarketSimulator(S0=S0, T=T, dt=dt)
    print(f"Simulating {n_simulations} paths...")
    prices = simulator.simulate_gbm(n_simulations, mu, sigma)

    # 2. Prepare Data
    print("Preparing data...")
    X, y = prepare_data(prices, strike, T, dt)

    # 3. Create Model
    # Features: 3
    model = create_deep_hedging_model(n_steps=n_steps, n_features=3)

    # 4. Compile
    loss_fn = DeepHedgingLoss(
        risk_aversion=risk_aversion,
        transaction_cost=transaction_cost,
        strike=strike
    )

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=loss_fn)

    # 5. Train
    print("Training model...")
    history = model.fit(
        X, y,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        verbose=1
    )

    return model, history, simulator
