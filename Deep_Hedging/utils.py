import numpy as np
from scipy.stats import norm

def bs_call_price(S, K, T, r, sigma):
    """
    Calculates the Black-Scholes price for a European Call Option.

    Args:
        S: Spot price
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free interest rate
        sigma: Volatility

    Returns:
        Call option price
    """
    # Handle edge case where T is 0 or close to 0
    if np.any(T <= 1e-8):
        # If T is 0, the price is max(S - K, 0)
        return np.maximum(S - K, 0.0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return price

def bs_call_delta(S, K, T, r, sigma):
    """
    Calculates the Black-Scholes Delta for a European Call Option.

    Args:
        S: Spot price
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free interest rate
        sigma: Volatility

    Returns:
        Call option delta
    """
    # Handle edge case where T is 0 or close to 0
    if np.any(T <= 1e-8):
        # At maturity, delta is 1 if S > K, 0 otherwise (approximated)
        return np.where(S > K, 1.0, 0.0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)
