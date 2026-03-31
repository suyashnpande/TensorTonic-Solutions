import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    N = len(seqs)
    if N==0:
        return np.array([])
        
    if max_len is None:
        max_len = max(len(seq) for seq in seqs) if seqs else 0

    padded = np.full((N, max_len), pad_value)
    
    for i, seq in (enumerate(seqs)):
        length = min(len(seq), max_len)
        padded[i, :length] = seq[:length]

    return padded