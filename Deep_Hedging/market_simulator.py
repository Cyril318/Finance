import numpy as np

class MarketSimulator:
    def __init__(self, S0=100.0, T=30/365, dt=1/365):
        """
        Initialize the Market Simulator.

        Args:
            S0: Initial stock price
            T: Time horizon in years
            dt: Time step size in years
        """
        self.S0 = S0
        self.T = T
        self.dt = dt

    def simulate_gbm(self, n_simulations, mu, sigma):
        """
        Simulates Geometric Brownian Motion paths.

        Args:
            n_simulations: Number of paths to simulate
            mu: Drift
            sigma: Volatility

        Returns:
            np.array of shape (n_simulations, n_steps + 1, 1) containing stock prices.
        """
        n_steps = int(np.round(self.T / self.dt))

        # Generate random normal variables
        # shape: (n_simulations, n_steps)
        Z = np.random.normal(0, 1, (n_simulations, n_steps))

        # Calculate drift and diffusion terms
        drift = (mu - 0.5 * sigma ** 2) * self.dt
        diffusion = sigma * np.sqrt(self.dt) * Z

        # Calculate log returns
        log_returns = drift + diffusion

        # Cumulative sum to get log prices
        # Prepend 0s for the initial state
        log_returns_cumulative = np.cumsum(log_returns, axis=1)
        log_returns_cumulative = np.concatenate(
            [np.zeros((n_simulations, 1)), log_returns_cumulative], axis=1
        )

        # Calculate prices
        S = self.S0 * np.exp(log_returns_cumulative)

        # Reshape to (n_simulations, n_steps + 1, 1)
        return S[..., np.newaxis]

    def simulate_heston(self, n_simulations, mu, kappa, theta, xi, rho, v0=None):
        """
        Simulates Heston Stochastic Volatility Model paths.

        Args:
            n_simulations: Number of paths
            mu: Drift of the asset
            kappa: Mean reversion speed of variance
            theta: Long-term variance
            xi: Volatility of volatility
            rho: Correlation between asset and variance Brownian motions
            v0: Initial variance (defaults to theta if None)

        Returns:
            np.array of shape (n_simulations, n_steps + 1, 1) containing stock prices.
        """
        n_steps = int(np.round(self.T / self.dt))

        if v0 is None:
            v0 = theta

        # Covariance matrix for correlated Brownian motions
        cov = np.array([[1, rho], [rho, 1]])
        # shape: (n_simulations, n_steps, 2)
        Z = np.random.multivariate_normal([0, 0], cov, (n_simulations, n_steps))

        Z_S = Z[:, :, 0] # Asset noise
        Z_v = Z[:, :, 1] # Variance noise

        S = np.zeros((n_simulations, n_steps + 1))
        v = np.zeros((n_simulations, n_steps + 1))

        S[:, 0] = self.S0
        v[:, 0] = v0

        for t in range(n_steps):
            # Euler-Maruyama discretization for variance (with Reflection for positivity)
            # dv = kappa * (theta - v) * dt + xi * sqrt(v) * dW_v

            # Current variance (ensure positive for sqrt)
            v_curr = np.maximum(v[:, t], 0)

            dv = kappa * (theta - v_curr) * self.dt + xi * np.sqrt(v_curr) * np.sqrt(self.dt) * Z_v[:, t]
            v[:, t+1] = v[:, t] + dv

            # Asset process
            # dS = mu * S * dt + sqrt(v) * S * dW_S

            # Use current variance for asset step
            vol_curr = np.sqrt(v_curr)

            dS = mu * S[:, t] * self.dt + vol_curr * S[:, t] * np.sqrt(self.dt) * Z_S[:, t]
            S[:, t+1] = S[:, t] + dS

        return S[..., np.newaxis]
