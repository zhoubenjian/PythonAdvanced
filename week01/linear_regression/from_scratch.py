"""A small, testable linear-regression implementation will live here."""
from __future__ import annotations

import numpy as np


def predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    '''
    线性回归预测
    :param X: 输入特征
    :param w: 权重
    :param b: 偏置
    :return: 预测值
    '''
    return X @ w + b


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    '''
    均方误差
    :param y_true: 真实值
    :param y_pred: 预测值
    :return: MSE
    '''
    return float(np.mean((y_true - y_pred) ** 2))


def fit(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    epochs: int = 1000
) -> (np.ndarray, float, list[float]):
    '''
    梯度下降算法
    :param X: 输入特征
    :param y: 目标变量
    :return: (w, b, loss_history)
    '''
    n_samples, n_features = X.shape

    w = np.zeros(n_features, dtype=float)
    b = 0.0
    losses: list[float] = []

    for epoch in range(epochs):
        y_pred = predict(X, w, b)
        error = y_pred - y

        # 对w求梯度（偏导数）
        dw = (2 / n_samples) * (X.T @ error)
        # 对b求梯度（偏导数）
        db = float((2 / n_samples) * np.sum(error))

        # 更新w和b
        w -= learning_rate * dw
        b -= learning_rate * db

        # 记录损失
        losses.append(mse(y, y_pred))

    return w, b, losses


def train() -> None:
    """Placeholder for the first week's implementation."""
    raise NotImplementedError
