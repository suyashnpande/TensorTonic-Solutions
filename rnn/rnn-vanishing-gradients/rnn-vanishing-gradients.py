import numpy as np

def compute_gradient_norm_decay(T: int, W_hh: np.ndarray) -> list:
    """
    Simulate gradient norm decay over T time steps.
    Returns list of gradient norms.
    """
    # YOUR CODE HERE
    hidden_size = W_hh.shape[0]

    dh = np.ones(hidden_size)

    norms = []

    for t in range(T):
        dh = W_hh.T @ dh
        norms.append(np.linalg.norm(dh))

    return norms
    pass