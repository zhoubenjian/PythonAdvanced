import numpy as np

from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


labels = np.array([1, 1, 1, 0, 0, 0])
preds = np.array([1, 0, 1, 1, 0, 0])


def test_metrics(
    y_true: NDArray[int],
    y_pred: NDArray[int],
) -> None:
    """测试指标函数。"""
    print(f'\naccuracy_score: {accuracy_score(y_true, y_pred):.4f}')
    print(f'precision_score: {precision_score(y_true, y_pred):.4f}')
    print(f'recall_score: {recall_score(y_true, y_pred):.4f}')
    print(f'f1_score: {f1_score(y_true, y_pred):.4f}')

