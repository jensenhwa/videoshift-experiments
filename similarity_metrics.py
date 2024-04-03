from enum import Enum
import numpy as np
import torch


class Similarity(Enum):
    DOT = 1
    COSINE = 2
    EUCLID = 3

    '''
    Enum for different available similarity metrics
    Args:
        a (np.array):               Shape = (A, embed_dim)
        b (np.array):               Shape = (B, embed_dim)
    Returns:
        similarities (np.array):    Shape = (A, B)
    '''

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self is Similarity.DOT:
            return torch.sum(a[:, None, :] * b[None, :, :], axis=2)

        if self is Similarity.COSINE:
            a_mag = torch.sqrt(torch.sum(torch.square(a), axis=1))
            b_mag = torch.sqrt(torch.sum(torch.square(b), axis=1))
            return torch.sum(a[:, None, :] * b[None, :, :], axis=2) / (a_mag[:, None] * b_mag[None, :])

        if self is Similarity.EUCLID:
            return -1 * torch.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
