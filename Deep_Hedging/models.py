import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, TimeDistributed
from tensorflow.keras.models import Model

def create_deep_hedging_model(n_steps, n_features, units=32, activation='sigmoid'):
    """
    Creates a Deep Hedging Keras model.

    Args:
        n_steps: Number of time steps (sequence length).
        n_features: Number of input features per step.
        units: Number of units in the recurrent layer.
        activation: Activation function for the output (delta).
                    'sigmoid' for [0, 1] constraint (Long Call hedging).
                    'linear' for unconstrained.

    Returns:
        A compiled Keras model.
    """
    # Input tensor shape: (batch_size, n_steps, n_features)
    inputs = Input(shape=(n_steps, n_features), name='market_data')

    # Recurrent layer
    # return_sequences=True ensures we get a delta for each time step
    x = LSTM(units, return_sequences=True)(inputs)
    # Alternatively use GRU: x = GRU(units, return_sequences=True)(inputs)

    # Dense layer to map to Delta
    # TimeDistributed applies the same Dense layer to every time step
    outputs = TimeDistributed(Dense(1, activation=activation), name='delta')(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model
