from __future__ import annotations

from typing import List

import numpy as np

from descents import BaseDescent
from descents import get_descent

from sklearn.metrics import r2_score

class LinearRegression:
    """
    Linear regression class
    """

    def __init__(self, descent_config: dict = {}, tolerance: float = 1e-4, max_iter: int = 300):
        """
        :param descent_config: gradient descent config
        :param tolerance: stopping criterion for square of euclidean norm of weight difference (float)
        :param max_iter: stopping criterion for iterations (int)
        """
        self.descent: BaseDescent = get_descent(descent_config)

        self.tolerance: float = tolerance
        self.max_iter: int = max_iter

        self.loss_history: List[float] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> LinearRegression:
        """
        Fitting descent weights for x and y dataset
        :param x: features array
        :param y: targets array
        :return: self
        """
        # TODO: fit weights to x and y
        self.i = 0
        self.loss_history.append(self.calc_loss(x, y))
        weights_diff = 10**10
        while self.i < self.max_iter and \
                not np.isnan(weights_diff).any() and \
                np.square(np.linalg.norm(weights_diff)) >= self.tolerance:
            weights_diff = self.descent.step(x, y)
            self.loss_history.append(self.calc_loss(x, y))
            self.i += 1
        # print(f'nan: {not np.isnan(weights_diff).any()}, toler: {np.square(np.linalg.norm(weights_diff))}')

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicting targets for x dataset
        :param x: features array
        :return: prediction: np.ndarray
        """
        return self.descent.predict(x)

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculating loss for x and y dataset
        :param x: features array
        :param y: targets array
        """
        return self.descent.calc_loss(x, y)

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(x)
        return r2_score(y, y_pred)
