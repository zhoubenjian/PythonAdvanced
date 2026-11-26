"""NumPy exercises for binary logistic regression."""

import numpy as np
from numpy.typing import NDArray


def sigmoid(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert logits to probabilities, preserving the input shape."""
    # 两支公式等价，分开计算可保证传给 exp 的值始终非正，避免溢出。
    probabilities = np.empty_like(logits, dtype=np.float64)
    nonnegative = logits >= 0
    probabilities[nonnegative] = 1 / (1 + np.exp(-logits[nonnegative]))
    exp_negative = np.exp(logits[~nonnegative])
    probabilities[~nonnegative] = exp_negative / (1 + exp_negative)
    return probabilities


def predict_proba(
    features: NDArray[np.float64],
    weights: NDArray[np.float64],
    intercept: float,
) -> NDArray[np.float64]:
    """Return one positive-class probability per sample."""
    # 每行对应一个样本，线性分数经 Sigmoid 转为正类概率。
    return sigmoid((features @ weights) + intercept)


def predict(
    features: NDArray[np.float64],
    weights: NDArray[np.float64],
    intercept: float,
    threshold: float = 0.5,
) -> NDArray[np.int_]:
    """Return 1 for probabilities at or above the threshold, otherwise 0."""
    # 概率恰好等于阈值时归为正类。
    return (predict_proba(features, weights, intercept) >= threshold).astype(int)


def binary_cross_entropy(
    y_true: NDArray[np.int_],
    y_pred: NDArray[np.float64],
) -> float:
    """Compute the binary cross-entropy loss."""
    # 将概率限制在开区间 (0, 1)，避免 log(0) 和 0 × (-inf)。
    probabilities = np.clip(y_pred, 1e-15, 1 - 1e-15)
    log_likelihood = y_true * np.log(probabilities) + (1 - y_true) * np.log1p(-probabilities)
    return float(-np.mean(log_likelihood))


def fit(
    features: NDArray[np.float64],
    labels: NDArray[np.int_],
    learning_rate: float = 0.1,
    epochs: int = 1000,
) -> tuple[NDArray[np.float64], float, list[float]]:
    """训练二分类逻辑回归，返回权重、截距和每轮更新前的 BCE。"""
    weights = np.zeros(features.shape[1])
    intercept = 0.0
    losses: list[float] = []

    for _ in range(epochs):
        logits = (features @ weights) + intercept
        probs = sigmoid(logits)

        # BCE 对 logit 的单样本导数为 p - y。
        error = probs - labels

        # 对样本梯度取平均，得到每个特征的权重梯度和标量截距梯度。
        dw = (features.T @ error) / len(labels)
        db = np.mean(error)

        weights -= dw * learning_rate
        intercept -= db * learning_rate

        # probs 由更新前的参数算出，初始记录对应零参数。
        losses.append(binary_cross_entropy(labels, probs))

    return weights, float(intercept), losses
