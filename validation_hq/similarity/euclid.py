import sklearn
import numpy as np
from similarity.base import Similarity

class Euclid(Similarity):
    @staticmethod
    def similarity(f, pair_indices):
        dist = np.linalg.norm(f[pair_indices[:, 0]] - f[pair_indices[:, 1]], axis=1)
        return (dist.max() - dist) / (dist.max() - dist.min() + 1e-10)

    def name(self):
        return "Euclid"
