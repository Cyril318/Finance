import tensorflow as tf

class DeepHedgingLoss(tf.keras.losses.Loss):
    def __init__(self, risk_aversion=1.0, transaction_cost=0.0, strike=100.0, name="deep_hedging_loss"):
        super().__init__(name=name)
        self.risk_aversion = risk_aversion
        self.transaction_cost = transaction_cost
        self.strike = strike

    def call(self, y_true, y_pred):
        """
        Computes the exponential utility loss.

        y_true: Tensor of shape (batch, n_steps + 1, 1) containing stock prices S.
                Note: The last step is needed for the final price change and payoff.
        y_pred: Tensor of shape (batch, n_steps, 1) containing deltas.
        """

        # Prices S: S_0, ..., S_T (n_steps + 1)
        # y_true shape might be (batch, n_steps + 1, 1)
        # y_pred shape is (batch, n_steps, 1)

        # S_t for t = 0 to T-1
        S_curr = y_true[:, :-1, :]
        # S_{t+1} for t = 0 to T-1 (which gives S_1 to S_T)
        S_next = y_true[:, 1:, :]

        # Price changes: dS = S_{t+1} - S_t
        dS = S_next - S_curr

        # Deltas
        deltas = y_pred

        # 1. PnL from holding the asset
        # sum(delta_t * (S_{t+1} - S_t))
        # shape: (batch, n_steps, 1)
        pnl_gross_per_step = deltas * dS
        pnl_gross = tf.reduce_sum(pnl_gross_per_step, axis=1) # (batch, 1)

        # 2. Transaction Costs
        # Cost_t = |delta_t - delta_{t-1}| * S_t * rate
        # We need delta_{t-1}. For t=0, delta_{-1} = 0.

        # Shift deltas to the right to get prev_deltas
        # Insert 0 at the beginning
        # shape (batch, 1, 1)
        zeros = tf.zeros_like(deltas[:, :1, :])
        prev_deltas = tf.concat([zeros, deltas[:, :-1, :]], axis=1)

        # Change in delta
        d_delta = deltas - prev_deltas

        # Transaction costs per step
        costs_per_step = tf.abs(d_delta) * S_curr * self.transaction_cost
        total_costs = tf.reduce_sum(costs_per_step, axis=1) # (batch, 1)

        # 3. Payoff at maturity (Short Call)
        # S_T is the last price in y_true
        S_T = y_true[:, -1, :]
        payoff = tf.maximum(S_T - self.strike, 0.0) # (batch, 1)

        # 4. Final Wealth (relative to initial wealth)
        # We want to Hedge, so we want Wealth_Final to cover Payoff.
        # Hedging Error = PnL - Payoff
        # Actually, if we are Short Call, our portfolio is:
        # Premium + PnL_hedging - Payoff
        # We maximize Utility(Premium + PnL - Payoff).
        # Since Premium is constant, we maximize Utility(PnL - Payoff).

        wealth = pnl_gross - total_costs - payoff

        # 5. Exponential Utility Loss
        # Loss = E[ exp(-lambda * Wealth) ]
        # We minimize this expectation.

        # Note: If lambda is 0 (risk neutral), we maximize E[Wealth].
        # Loss = - E[Wealth]
        # But exp(0) is 1, so the formula needs care for lambda=0 case if strictly implemented.
        # The prompt says lambda=0 : Risk Neutral.
        # Limit lambda->0 of (1 - exp(-lambda*W))/lambda is W.
        # But simpler: if lambda > 0 use exp, if lambda approx 0 use -mean(wealth).

        # However, typically lambda is small but non-zero.
        # Let's handle lambda close to 0 separately or just use the formula if lambda > 1e-6.

        if self.risk_aversion > 1e-6:
            loss = tf.reduce_mean(tf.exp(-self.risk_aversion * wealth))
        else:
            loss = -tf.reduce_mean(wealth)

        return loss
