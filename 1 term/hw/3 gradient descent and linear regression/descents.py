from dataclasses import dataclass
from enum import auto
from enum import Enum
from typing import Dict
from typing import Type, Tuple

import numpy as np


@dataclass
class LearningRate:
    lambda_: float = 1e-3
    s0: float = 1
    p: float = 0.5

    iteration: int = 0

    def __call__(self):
        """
        Calculate learning rate according to lambda (s0/(s0 + t))^p formula
        """
        self.iteration += 1
        return self.lambda_ * (self.s0 / (self.s0 + self.iteration)) ** self.p


class LossFunction(Enum):
    MSE = auto()
    MAE = auto()
    LogCosh = auto()
    Huber = auto()


class BaseDescent:
    """
    A base class and templates for all functions
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE, delta: float = 1):
        """
        :param dimension: feature space dimension
        :param lambda_: learning rate parameter
        :param loss_function: optimized loss function
        """
        self.w: np.ndarray = np.random.rand(dimension)
        self.lr: LearningRate = LearningRate(lambda_=lambda_)
        self.loss_function: LossFunction = loss_function
        self.delta = delta

    def step(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.update_weights(self.calc_gradient(x, y))

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Template for update_weights function
        Update weights with respect to gradient
        :param gradient: gradient
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        pass

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Template for calc_gradient function
        Calculate gradient of loss function with respect to weights
        :param x: features array
        :param y: targets array
        :return: gradient: np.ndarray
        """
        pass

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate loss for x and y with our weights
        :param x: features array
        :param y: targets array
        :return: loss: float
        """
        y = y.squeeze()
        if self.loss_function == LossFunction.MSE:
            loss = np.square(y - x @ self.w).mean()
        elif self.loss_function == LossFunction.MAE:
            loss = np.abs(y - x @ self.w).mean()
        elif self.loss_function == LossFunction.LogCosh:
            arg = (y - x @ self.w).squeeze()
            loss = (np.logaddexp(arg, -arg) - np.log(2)).mean()
        elif self.loss_function == LossFunction.Huber:
            a = y - x @ self.w
            mse_loss = np.square(a)
            mae_loss = np.abs(a)
            loss = ((mae_loss <= self.delta) * mse_loss + (mae_loss > self.delta) * mae_loss).mean()
        # print(y.shape, x.shape, self.w.shape, (x @ self.w).shape)
        return loss

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate predictions for x
        :param x: features array
        :return: prediction: np.ndarray
        """
        return x @ self.w
        


class VanillaGradientDescent(BaseDescent):
    """
    Full gradient descent class
    """

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        delta = -self.lr() * gradient
        self.w += delta
        return delta

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # y = y.squeeze()
        # print(x.shape[0])
        if self.loss_function == LossFunction.MSE:
            grad = -2 * ((y - x @ self.w[:, np.newaxis]).T @ x).squeeze()
        elif self.loss_function == LossFunction.MAE:
            grad = - ((np.sign(y - x @ self.w[:, np.newaxis])).T @ x).squeeze()
        elif self.loss_function == LossFunction.LogCosh:
            grad = - ((np.tanh(y - x @ self.w[:, np.newaxis])).T @ x).squeeze()
        elif self.loss_function == LossFunction.Huber:
            a = y - x @ self.w[:, np.newaxis]
            mse_grad = - ((a * (np.abs(a) <= self.delta)).T @ x).squeeze()
            mae_grad = - self.delta * ((np.sign(a * (np.abs(a) > self.delta))).T @ x).squeeze()
            grad = mse_grad + mae_grad
        # print(f'sum grad: {np.square(np.linalg.norm(grad))}')
        # print(f'grad: {np.square(np.linalg.norm(grad / x.shape[0]))}')
        return grad / x.shape[0]


class StochasticDescent(VanillaGradientDescent):
    """
    Stochastic gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, batch_size: int = 50,
                 loss_function: LossFunction = LossFunction.MSE):
        """
        :param batch_size: batch size (int)
        """
        super().__init__(dimension, lambda_, loss_function)
        self.batch_size = batch_size

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x_batch, y_batch = next(self._batch_generator(x, y))
        # print(x_batch.shape)
        return super().calc_gradient(x_batch, y_batch)

    def _batch_generator(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        while True:
            self.indeces = np.random.randint(0, x.shape[0], size=x.shape[1])
            for i in range(0, x.shape[0] - self.batch_size + 1, self.batch_size):
                batch_ind = self.indeces[i:i + self.batch_size]
                yield x[batch_ind], y[batch_ind]


class MomentumDescent(VanillaGradientDescent):
    """
    Momentum gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.alpha: float = 0.9

        self.h: np.ndarray = np.zeros(dimension)

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        
        self.h =  self.alpha * self.h + self.lr() * gradient
        self.w += -self.h
        return -self.h


class Adam(VanillaGradientDescent):
    """
    Adaptive Moment Estimation gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.eps: float = 1e-8

        self.m: np.ndarray = np.zeros(dimension)
        self.v: np.ndarray = np.zeros(dimension)

        self.beta_1: float = 0.9
        self.beta_2: float = 0.999

        self.iteration: int = 0

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        self.iteration += 1
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * gradient
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * gradient ** 2
        alpha = self.lr() * np.sqrt(1 - self.beta_2**self.iteration) / (1 - self.beta_1**self.iteration)
        delta = - alpha * self.m / (np.sqrt(self.v) + self.eps)
        self.w += delta
        return delta


class AdaMax(Adam):
    """
    Adam Estimation with infinite norm gradient descent class
    """

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        self.iteration += 1
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * gradient
        self.v = np.max(np.stack((self.beta_2 * self.v, np.abs(gradient))), axis=0)
        alpha = self.lr() * np.sqrt(1 - self.beta_2**self.iteration) / (1 - self.beta_1**self.iteration)
        delta = - self.lr() / (1 - self.beta_1**self.iteration) * self.m / (self.v + self.eps)
        self.w += delta
        return delta

class BaseDescentReg(BaseDescent):
    """
    A base class with regularization
    """

    def __init__(self, *args, mu: float = 0, **kwargs):
        """
        :param mu: regularization coefficient (float)
        """
        super().__init__(*args, **kwargs)

        self.mu = mu

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate gradient of loss function and L2 regularization with respect to weights
        """
        
        l2_gradient: np.ndarray = self.w

        return super().calc_gradient(x, y) + l2_gradient * self.mu


class VanillaGradientDescentReg(BaseDescentReg, VanillaGradientDescent):
    """
    Full gradient descent with regularization class
    """


class StochasticDescentReg(BaseDescentReg, StochasticDescent):
    """
    Stochastic gradient descent with regularization class
    """


class MomentumDescentReg(BaseDescentReg, MomentumDescent):
    """
    Momentum gradient descent with regularization class
    """


class AdamReg(BaseDescentReg, Adam):
    """
    Adaptive gradient algorithm with regularization class
    """


def get_descent(descent_config: dict) -> BaseDescent:
    descent_name = descent_config.get('descent_name', 'full')
    regularized = descent_config.get('regularized', False)

    descent_mapping: Dict[str, Type[BaseDescent]] = {
        'full': VanillaGradientDescent if not regularized else VanillaGradientDescentReg,
        'stochastic': StochasticDescent if not regularized else StochasticDescentReg,
        'momentum': MomentumDescent if not regularized else MomentumDescentReg,
        'adam': Adam if not regularized else AdamReg,
        'adamax': AdaMax
    }

    if descent_name not in descent_mapping:
        raise ValueError(f'Incorrect descent name, use one of these: {descent_mapping.keys()}')

    descent_class = descent_mapping[descent_name]

    return descent_class(**descent_config.get('kwargs', {}))
