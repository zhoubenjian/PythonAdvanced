"""Compare the from-scratch implementation with scikit-learn."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression

from week01.linear_regression.from_scratch import fit, mse, predict


@dataclass(frozen=True)
class ModelResult:
    """Parameters and training error produced by a linear model."""

    weights: NDArray[np.float64]
    intercept: float
    mse: float


def create_dataset() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Create a deterministic dataset following y = 2x + 1."""
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = 2 * X[:, 0] + 1
    return X, y


def train_from_scratch(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
) -> ModelResult:
    """Train the gradient-descent implementation and return its result."""
    weights, intercept, _ = fit(X, y)
    y_pred = predict(X, weights, intercept)
    return ModelResult(weights, intercept, mse(y, y_pred))


def train_with_sklearn(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
) -> ModelResult:
    """Train scikit-learn's least-squares model and return its result."""
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    return ModelResult(model.coef_, float(model.intercept_), mse(y, y_pred))


def compare_models() -> None:
    """Train both models and print a concise comparison."""
    X, y = create_dataset()
    scratch = train_from_scratch(X, y)
    sklearn = train_with_sklearn(X, y)

    print(
        f"手写模型: 系数 w={scratch.weights}, 截距 b={scratch.intercept:.8f}, "
        f"MSE={scratch.mse:.6e}"
    )
    print(
        f"sklearn: 系数 w={sklearn.weights}, 截距 b={sklearn.intercept:.8f}, "
        f"MSE={sklearn.mse:.6e}"
    )


if __name__ == "__main__":
    compare_models()
