"""Acceptance tests for the Week 02 prediction exercises."""

import numpy as np

from week02.logistic_regression.from_scratch import (
    binary_cross_entropy,
    predict,
    predict_proba,
    sigmoid,
)


def test_sigmoid() -> None:
    probabilities = sigmoid(np.array([-2.0, 0.0, 2.0]))

    assert probabilities.shape == (3,)
    np.testing.assert_allclose(probabilities[1], 0.5)
    assert 0 < probabilities[0] < probabilities[1] < probabilities[2] < 1


def test_predict_proba() -> None:
    features = np.array([[-1.0], [0.0], [1.0]])
    probabilities = predict_proba(features, np.array([2.0]), 1.0)

    assert probabilities.shape == (3,)
    np.testing.assert_allclose(probabilities, [0.26894142, 0.73105858, 0.95257413])


def test_sigmoid_extreme_logits() -> None:
    with np.errstate(over="raise"):
        probabilities = sigmoid(np.array([-1000.0, 0.0, 1000.0]))

    np.testing.assert_allclose(probabilities, [0.0, 0.5, 1.0])


def test_binary_cross_entropy() -> None:
    labels = np.array([1, 0])

    np.testing.assert_allclose(binary_cross_entropy(labels, np.array([0.9, 0.1])), 0.10536052)
    np.testing.assert_allclose(binary_cross_entropy(labels, np.array([0.1, 0.9])), 2.30258509)


def test_binary_cross_entropy_boundary_probabilities() -> None:
    labels = np.array([1, 0])

    with np.errstate(divide="raise", invalid="raise"):
        correct = binary_cross_entropy(labels, np.array([1.0, 0.0]))
        incorrect = binary_cross_entropy(labels, np.array([0.0, 1.0]))

    assert np.isfinite(correct) and np.isfinite(incorrect)
    assert 0 <= correct < incorrect


def test_predict() -> None:
    features = np.array([[-2.0], [0.0], [2.0]])
    weights = np.array([1.0])

    # 概率恰好等于阈值时，按约定归入正类。
    np.testing.assert_array_equal(predict(features, weights, 0.0), [0, 1, 1])
    np.testing.assert_array_equal(
        predict(features, weights, 0.0, threshold=0.8),
        [0, 0, 1],
    )
