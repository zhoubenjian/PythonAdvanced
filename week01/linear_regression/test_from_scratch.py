"""Tests for the from-scratch implementation."""
import numpy as np

from week01.linear_regression.from_scratch import fit, mse, predict


def test_predict() -> None:
    # 用已知公式构造期望值，验证矩阵预测没有改变输入含义。
    X = np.array([[1.0], [2.0], [3.0]])
    w = np.array([2.0])
    b = 1.0

    result = predict(X, w, b)
    np.testing.assert_array_equal(result, np.array([3.0, 5.0, 7.0]))


def test_mse() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 3.0, 2.0])

    # 浮点运算存在精度误差，使用近似比较而不是直接比较相等。
    assert np.isclose(mse(y_true, y_pred), 2 / 3)


def test_fit() -> None:
    X = np.array([
        [1.0],
        [2.0],
        [3.0],
        [4.0],
    ])
    y = 2 * X[:, 0] + 1

    w, b, losses = fit(
        X,
        y,
        learning_rate=0.01,
        epochs=2000,
    )

    # 允许有限迭代带来的近似误差，同时验证损失确实下降。
    np.testing.assert_allclose(w, [2.0], atol=0.05)
    assert abs(b - 1.0) < 0.05
    assert losses[-1] < losses[0]
