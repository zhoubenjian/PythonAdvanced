"""Tests for the from-scratch and scikit-learn comparison."""

import numpy as np

from week01.linear_regression.sklearn_baseline import (
    create_dataset,
    train_from_scratch,
    train_with_sklearn,
)


def test_from_scratch_result() -> None:
    X, y = create_dataset()

    result = train_from_scratch(X, y)

    np.testing.assert_allclose(result.weights, [2.0], atol=0.05)
    assert abs(result.intercept - 1.0) < 0.05
    assert result.mse < 0.001


def test_sklearn_result() -> None:
    X, y = create_dataset()

    result = train_with_sklearn(X, y)

    np.testing.assert_allclose(result.weights, [2.0])
    assert abs(result.intercept - 1.0) < 1e-10
    assert result.mse < 1e-10
