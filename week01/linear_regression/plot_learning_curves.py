import matplotlib.pyplot as plt

from week01.linear_regression.from_scratch import fit
from week01.linear_regression.sklearn_baseline import create_dataset


def plot_learning_curves() -> None:

    X, y = create_dataset()
    learning_rates = [0.001, 0.01, 0.05, 0.1, 0.2]

    for learning_rate in learning_rates:
        _, _, losses = fit(
            X,
            y,
            learning_rate=learning_rate,
            epochs=100,
        )

        epochs = range(1, len(losses) + 1)
        plt.plot(epochs, losses, label=f'learning_rate={learning_rate}')

    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.title("Learning Rate Comparison")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    plot_learning_curves()
