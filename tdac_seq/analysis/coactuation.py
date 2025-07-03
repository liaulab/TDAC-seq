import numpy as np

def calc_coactuation(dad: np.ndarray) -> np.ndarray:
    """
    Returns the fraction of reads where the row position and column position are both in a DAD. `dad` is a binary matrix where 1 indicates a DAD and 0 indicates a non-DAD with shape (sequence x reads). Output is a square matrix with shape (sequence x sequence).
    """
    return (dad.astype(float).T @ dad.astype(float)) / dad.shape[0] # (sequence x reads) @ (reads x sequence) = sequence x sequence
