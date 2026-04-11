import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    """
    Forward pass through entire sequence.
    """
    # YOUR CODE HERE
    time = X.shape[1]
    output_ = []
    h = h_0
    for t in range(time):
        h = np.tanh( np.dot(X[:,t,:], W_xh.T) + np.dot(h_0, W_hh.T) + b_h)
        output_.append(h)

    outputs = np.stack(output_, axis=1)  # (batch, time, hidden)
    return outputs, h
