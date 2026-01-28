import numpy as np
import matplotlib.pyplot as plt
import os
from Deep_Hedging.trainer import train_hedging_model
from Deep_Hedging.market_simulator import MarketSimulator
from Deep_Hedging.utils import bs_call_delta
from Deep_Hedging.trainer import prepare_data

def evaluate_model(model, prices, strike, T, dt):
    """
    Evaluates the model on given prices.
    Returns Deltas.
    """
    X, _ = prepare_data(prices, strike, T, dt)
    deltas = model.predict(X, batch_size=256, verbose=0)
    return deltas

def calculate_pnl(prices, deltas, strike, transaction_cost):
    """
    Calculates PnL for a given strategy.
    """
    # prices: (n_sims, n_steps + 1, 1)
    # deltas: (n_sims, n_steps, 1)

    S_curr = prices[:, :-1, 0]
    S_next = prices[:, 1:, 0]
    dS = S_next - S_curr

    delta_vals = deltas[:, :, 0]

    # Gross PnL
    pnl_gross = np.sum(delta_vals * dS, axis=1)

    # Transaction Costs
    # Prepend 0 to deltas
    prev_deltas = np.concatenate([np.zeros((deltas.shape[0], 1)), delta_vals[:, :-1]], axis=1)
    d_delta = delta_vals - prev_deltas
    costs = np.sum(np.abs(d_delta) * S_curr * transaction_cost, axis=1)

    # Payoff
    S_T = prices[:, -1, 0]
    payoff = np.maximum(S_T - strike, 0.0)

    # Final Wealth (PnL - Costs - Payoff)
    # This represents the hedging error (plus premium if we added it)
    wealth = pnl_gross - costs - payoff

    return wealth

def main():
    # Parameters
    S0 = 100
    K = 100
    T = 30/365
    dt = 1/365
    sigma = 0.2
    mu = 0.0

    # Training parameters
    n_train_sims = 50000 # Increase to 1M for better results
    epochs = 5

    # Create output directory for plots
    os.makedirs("results", exist_ok=True)

    # ==========================================
    # Phase 4: Sanity Check (No Frictions)
    # ==========================================
    print("\n--- Phase 4: Sanity Check (No Frictions) ---")
    model_no_cost, history_no_cost, _ = train_hedging_model(
        n_simulations=n_train_sims,
        n_steps=int(np.round(T/dt)),
        S0=S0, strike=K, T=T, mu=mu, sigma=sigma,
        risk_aversion=1.0,
        transaction_cost=0.0,
        epochs=epochs
    )

    # Verify against BS
    # Generate test data (small set)
    sim = MarketSimulator(S0=S0, T=T, dt=dt)
    test_prices_gbm = sim.simulate_gbm(1000, mu, sigma)

    # Predict
    deltas_ai = evaluate_model(model_no_cost, test_prices_gbm, K, T, dt)

    # Calculate BS Deltas
    # We need to compute BS Delta at each step
    # TTM at step t: T - t*dt
    n_steps = test_prices_gbm.shape[1] - 1
    times = np.arange(n_steps) * dt
    ttm = T - times
    ttm = np.maximum(ttm, 1e-6) # Avoid division by zero

    # Broadcasting BS calculation
    # prices shape (1000, 31, 1). We use first 30 prices.
    S_t = test_prices_gbm[:, :-1, 0]

    bs_deltas = np.zeros_like(S_t)
    for t in range(n_steps):
        bs_deltas[:, t] = bs_call_delta(S_t[:, t], K, ttm[t], mu, sigma) # mu=r=0 here

    # Plot Comparison
    plt.figure(figsize=(10, 6))

    # Flatten for scatter plot
    plt.scatter(S_t.flatten(), bs_deltas.flatten(), label='Black-Scholes', s=1, alpha=0.5)
    plt.scatter(S_t.flatten(), deltas_ai.flatten(), label='Deep Hedging (No Cost)', s=1, alpha=0.5, color='orange')
    plt.title("Phase 4: Delta vs Spot (Sanity Check)")
    plt.xlabel("Spot Price")
    plt.ylabel("Delta")
    plt.legend()
    plt.savefig("results/phase4_sanity_check.png")
    print("Saved Phase 4 plot to results/phase4_sanity_check.png")

    # ==========================================
    # Phase 5: Frictions (Transaction Costs)
    # ==========================================
    print("\n--- Phase 5: Frictions (Transaction Costs) ---")
    cost_rate = 0.002 # 20 bps
    model_cost, history_cost, _ = train_hedging_model(
        n_simulations=n_train_sims,
        n_steps=int(np.round(T/dt)),
        S0=S0, strike=K, T=T, mu=mu, sigma=sigma,
        risk_aversion=1.0,
        transaction_cost=cost_rate,
        epochs=epochs
    )

    # Predict
    deltas_ai_cost = evaluate_model(model_cost, test_prices_gbm, K, T, dt)

    # Plot "No-Action Band"
    plt.figure(figsize=(10, 6))
    plt.scatter(S_t.flatten(), bs_deltas.flatten(), label='Black-Scholes', s=1, alpha=0.1, color='blue')
    plt.scatter(S_t.flatten(), deltas_ai_cost.flatten(), label=f'Deep Hedging (Cost {cost_rate})', s=1, alpha=0.5, color='red')
    plt.title("Phase 5: Delta vs Spot (With Frictions)")
    plt.xlabel("Spot Price")
    plt.ylabel("Delta")
    plt.legend()
    plt.savefig("results/phase5_frictions.png")
    print("Saved Phase 5 plot to results/phase5_frictions.png")

    # ==========================================
    # Phase 6: Analysis (PnL & Heston)
    # ==========================================
    print("\n--- Phase 6: Analysis ---")

    # 1. PnL Distribution Comparison (on GBM data)
    # Calculate PnL for BS Strategy with Costs
    # We use BS Deltas computed earlier, but apply costs
    # Note: BS deltas need to be shaped (1000, 30, 1)
    bs_deltas_reshaped = bs_deltas[..., np.newaxis]

    pnl_bs = calculate_pnl(test_prices_gbm, bs_deltas_reshaped, K, cost_rate)
    pnl_ai = calculate_pnl(test_prices_gbm, deltas_ai_cost, K, cost_rate)

    plt.figure(figsize=(10, 6))
    plt.hist(pnl_bs, bins=50, alpha=0.5, label='Black-Scholes (with costs)', density=True)
    plt.hist(pnl_ai, bins=50, alpha=0.5, label='Deep Hedging', density=True)
    plt.title("PnL Distribution Comparison")
    plt.xlabel("PnL (Wealth)")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig("results/phase6_pnl_dist.png")
    print("Saved PnL distribution plot.")

    print(f"BS Mean PnL: {np.mean(pnl_bs):.4f}, Std: {np.std(pnl_bs):.4f}")
    print(f"AI Mean PnL: {np.mean(pnl_ai):.4f}, Std: {np.std(pnl_ai):.4f}")

    # 2. Robustness Test on Heston
    print("Testing on Heston Data...")
    test_prices_heston = sim.simulate_heston(1000, mu=mu, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)

    deltas_ai_heston = evaluate_model(model_cost, test_prices_heston, K, T, dt)
    pnl_ai_heston = calculate_pnl(test_prices_heston, deltas_ai_heston, K, cost_rate)

    # Compare with BS on Heston (BS assumes constant vol, so it's misspecified)
    S_t_heston = test_prices_heston[:, :-1, 0]
    bs_deltas_heston = np.zeros_like(S_t_heston)
    for t in range(n_steps):
        bs_deltas_heston[:, t] = bs_call_delta(S_t_heston[:, t], K, ttm[t], mu, sigma)

    pnl_bs_heston = calculate_pnl(test_prices_heston, bs_deltas_heston[..., np.newaxis], K, cost_rate)

    plt.figure(figsize=(10, 6))
    plt.hist(pnl_bs_heston, bins=50, alpha=0.5, label='Black-Scholes (on Heston)', density=True)
    plt.hist(pnl_ai_heston, bins=50, alpha=0.5, label='Deep Hedging (on Heston)', density=True)
    plt.title("Robustness Test: Heston Model")
    plt.xlabel("PnL")
    plt.legend()
    plt.savefig("results/phase6_heston_test.png")
    print("Saved Heston test plot.")

    print(f"BS Heston Mean PnL: {np.mean(pnl_bs_heston):.4f}, Std: {np.std(pnl_bs_heston):.4f}")
    print(f"AI Heston Mean PnL: {np.mean(pnl_ai_heston):.4f}, Std: {np.std(pnl_ai_heston):.4f}")

if __name__ == "__main__":
    main()
