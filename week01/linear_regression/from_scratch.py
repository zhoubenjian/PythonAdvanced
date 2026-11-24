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
    # 矩阵乘法同时计算所有样本的线性组合，避免逐样本 Python 循环。
    return X @ w + b


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    '''
    均方误差
    :param y_true: 真实值
    :param y_pred: 预测值
    :return: MSE
    '''
    # 先平方再取平均，避免正负误差相互抵消；平方也会放大大误差。
    return float(np.mean((y_true - y_pred) ** 2))


def fit(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    epochs: int = 1000
) -> tuple[np.ndarray, float, list[float]]:
    '''
    梯度下降算法
    :param X: 输入特征
    :param y: 目标变量
    :return: (w, b, loss_history)
    '''
    # w 的长度等于特征数；同一个特征在所有样本中共享一个权重。
    n_samples, n_features = X.shape

    # 从零参数开始，记录每轮更新前计算出的损失用于观察收敛过程。
    w = np.zeros(n_features, dtype=float)
    b = 0.0
    losses: list[float] = []

    for _ in range(epochs):
        y_pred = predict(X, w, b)
        error = y_pred - y

        # dw/db 是损失对参数的梯度，不是参数本身的变化量。
        dw = (2 / n_samples) * (X.T @ error)
        db = float((2 / n_samples) * np.sum(error))

        # 负梯度方向让损失下降，learning_rate 控制每次移动的步长。
        w -= learning_rate * dw
        b -= learning_rate * db

        # 当前实现记录的是更新前预测对应的损失，便于比较每轮输入状态。
        losses.append(mse(y, y_pred))

    return w, b, losses


def train() -> None:
    """Placeholder for the first week's implementation."""
    raise NotImplementedError
