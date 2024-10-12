from abc import ABCMeta, abstractmethod

import numpy as np
from scipy import sparse


class FeatureAugmentation(metaclass=ABCMeta):
    @property
    @abstractmethod
    def feature_dim(self) -> int:
        raise NotImplementedError

    def __call__(self, m: sparse.coo_matrix, b: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.degree})"


class ArnoldiAugmentation(FeatureAugmentation):
    def __init__(self, degree: int):
        super().__init__()
        self.degree = degree

    @property
    def feature_dim(self) -> int:
        return self.degree

    def __call__(self, m: sparse.coo_matrix, b: np.ndarray) -> np.ndarray:
        features = []
        v = b / np.linalg.norm(b)
        for _ in range(self.degree):
            v = m.dot(v)
            v = v / np.linalg.norm(v, ord=np.inf)
            features.append(v)
        features = np.stack(features, axis=-1)
        return features


class JacobiAugmentation(FeatureAugmentation):
    def __init__(self, degree: int):
        super().__init__()
        self.degree = degree

    @property
    def feature_dim(self) -> int:
        return self.degree + 1

    def __call__(self, m: sparse.coo_matrix, b: np.ndarray) -> np.ndarray:
        diagonal = m.diagonal()
        bias = b / diagonal
        features = [bias]
        diagonal_matrix = sparse.dia_matrix(
            (diagonal[np.newaxis, :], [0]), shape=m.shape
        )
        inverse_diagonal_matrix = sparse.dia_matrix(
            (1.0 / diagonal[np.newaxis, :], [0]), shape=m.shape
        )
        n_matrix = m - diagonal_matrix
        h_matrix = inverse_diagonal_matrix.dot(n_matrix)
        v = bias
        for _ in range(self.degree):
            v = h_matrix.dot(v) + bias
            features.append(v)
        features = np.stack(features, axis=-1)
        features = features / np.linalg.norm(
            features, ord=np.inf, axis=0, keepdims=True
        )
        return features


class ConjugateGradientAugmentation(FeatureAugmentation):
    def __init__(self, degree: int):
        super().__init__()
        self.degree = degree

    @property
    def feature_dim(self) -> int:
        return self.degree

    def __call__(self, m: sparse.coo_matrix, b: np.ndarray) -> np.ndarray:
        v = np.zeros_like(b)
        r = b.copy()
        p = r.copy()
        r_squared_norm = np.square(r).sum()
        features = []
        for _ in range(self.degree):
            Ap = m.dot(p)
            alpha = r_squared_norm / (p * Ap).sum()
            v = v + alpha * p
            r = r - alpha * Ap
            r1_squared_norm = np.square(r).sum()
            beta = r1_squared_norm / r_squared_norm
            p = r + beta * p
            r_squared_norm = r1_squared_norm
            features.append(v)
        features = np.stack(features, axis=-1)
        features = features / np.linalg.norm(
            features, ord=np.inf, axis=0, keepdims=True
        )
        return features

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.degree})"
