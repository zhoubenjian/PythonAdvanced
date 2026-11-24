import numpy as np

from week01.linear_regression.from_scratch import fit
from week01.linear_regression.sklearn_baseline import create_dataset


def run_experiment() -> None:
    # 所有学习率使用同一份数据，确保实验中只有学习率这个变量发生变化。
    X, y = create_dataset()

    learning_rates = [0.001, 0.01, 0.05, 0.1, 0.2]
    # 用统一的损失阈值衡量不同学习率的收敛速度。
    threshold = 0.01

    for learning_rate in learning_rates:
        _, _, losses = fit(
            X,
            y,
            learning_rate=learning_rate,
            epochs=100,
        )

        # 找到损失首次低于阈值的训练轮次；未达到阈值时返回 None。
        threshold_epoch = next(
            (
                epoch
                for epoch, loss in enumerate(losses, start=1)
                if loss < threshold
            ),
            None,
        )

        # 先排除 nan/inf；最终损失高于初始损失时，将本次实验视为发散。
        finite = np.all(np.isfinite(losses))
        diverged = not finite or losses[-1] > losses[0]

        print(
            f"learning_rate={learning_rate:.3f} | "
            f"initial_loss={losses[0]:.6e} | "
            f"final_loss={losses[-1]:.6e} | "
            f"min_loss={np.min(losses):.6e} | "
            f"threshold_epoch={threshold_epoch} | "
            f"finite={finite} | "
            f"diverged={diverged}"
        )


if __name__ == "__main__":
    run_experiment()
