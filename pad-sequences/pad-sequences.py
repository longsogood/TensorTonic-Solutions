import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    if not seqs:
        return np.empty()
        
    N = len(seqs)
    L = max([len(seq) for seq in seqs]) if not max_len else max_len
    output = np.full((N,L), pad_value)
    
    for row_index in range(output.shape[0]):
        if len(seqs[row_index]) >= L:
            output[row_index,:] = seqs[row_index][:L]
        else:
            output[row_index,:len(seqs[row_index])] = seqs[row_index]
    return output