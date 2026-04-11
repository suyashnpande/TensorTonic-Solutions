import numpy as np

class VanillaRNN:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim
        # Xavier initialization
        self.W_xh = np.random.randn(hidden_dim, input_dim)  * np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / (2 * hidden_dim))
        self.W_hy = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_h  = np.zeros(hidden_dim)
        self.b_y  = np.zeros(output_dim)

    def forward(self, X: np.ndarray, h_0: np.ndarray = None) -> tuple:
        N, T, input_dim = X.shape                 

        if h_0 is None:
            h_0 = np.zeros((N, self.hidden_dim))  
        h = h_0                                     
        hidden_states = []                    

        for t in range(T):
            x_t = X[:, t, :]                       
            h = np.tanh(x_t @ self.W_xh.T + h @ self.W_hh.T + self.b_h) 
            hidden_states.append(h)                 

        hidden_states = np.stack(hidden_states, axis=1)   # (N, T, H)

        # Reshape to (N*T, H) → multiply → reshape back to (N, T, output_dim)
        N_T = N * T
        h_flat = hidden_states.reshape(N_T, self.hidden_dim)       # (N*T, H)
        y_flat = h_flat @ self.W_hy.T + self.b_y                   # (N*T, output_dim)
        Y = y_flat.reshape(N, T, -1)                               # (N, T, output_dim)

        # h_final = last hidden state
        h_final = hidden_states[:, -1, :]                          # (N, H)

        return Y, h_final