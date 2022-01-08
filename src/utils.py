"""Utils."""

import numpy as np


def argmax(array):
    """Return the index of the maximum value in an array."""
    max_ = max(array)
    return np.random.choice([i for i, x in enumerate(array) if x == max_])


def vprint(vector: np.ndarray, n: int = 4) -> None:
    """Print a vector"""
    for i in range(0, len(vector), n):
        print(vector[i : i + n])


def print_matrix_stats(v: np.ndarray):
    """Print stats about the matrix"""
    print(f"Shape: {v.shape}")
    print(f"Min: {v.min()}")
    print(f"Max: {v.max()}")
    print(f"Mean: {v.mean()}")
    print(f"Std: {v.std()}")
    print(f"Sum: {v.sum()}")
    print("\n" * 2)
