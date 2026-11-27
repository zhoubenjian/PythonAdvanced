import numpy as np

from numpy.typing import NDArray


labels = np.array([1, 1, 1, 0, 0, 0])
preds = np.array([1, 0, 1, 1, 0, 0])


def manual_confusion_matrix(
    y_true: NDArray[int],
    y_pred: NDArray[int],
) -> None:
    """测试混淆矩阵函数。"""
    tp, fp, fn, tn = 0, 0, 0, 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            tp += 1
        elif t == 1 and p == 0:
            fn += 1
        elif t == 0 and p == 1:
            fp += 1
        else:
            tn += 1

    print(f'tp(真正例): {tp}, fp(假正例): {fp}, fn(假负例): {fn}, tn(真负例): {tn}\n')

    # 准确率（Accuracy）：所有预测正确的样本占总样本的比例
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    print(f'accuracy(准确率)：{accuracy:.4f}')

    # 精确率（Precision）：模型预测为正类的样本中，实际为正类的比例（强调减少误报 FP）
    precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
    print(f'precision(精确率)：{precision:.4f}')

    # 召回率（Recall）：实际为正类的样本中，被模型正确预测出来的比例（强调减少漏报 FN）
    recall = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
    print(f'recall(召回率)：{recall:.4f}')

    # F1分数（F1-Score）：精确率和召回率的调和平均数，用于综合评估模型在两者之间的平衡
    f1_score = 0.0 if (precision == 0 and recall == 0) else (2 * precision * recall) / (precision + recall)
    print(f'f1_score：(调和平均值)：{f1_score:.4f}')


if __name__ == '__main__':
    manual_confusion_matrix(labels, preds)
