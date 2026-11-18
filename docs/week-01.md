# 第 1 周：建立可复现的 Python 基线

## 目标

完成一个可以被测试、运行、打包和解释的最小机器学习项目骨架。

## 任务清单

- [ ] 确认当前本地 Python、NumPy、pandas、scikit-learn 和 pytest 版本，并记录到 README。
- [ ] 在现有本地环境中建立 `src/`、`tests/`、`notebooks/`、`data/`、`docs/`。
- [ ] 确认现有 lint、格式化和测试命令；若尚未配置，再补充 ruff 和 pytest。
- [ ] 用 NumPy 实现线性回归：损失函数、梯度下降、训练曲线。
- [ ] 用 scikit-learn 在同一数据集上复现，比较参数、耗时和指标。
- [ ] 写一页实验记录：假设、数据切分、结果、失败原因、下一步。

## 验收标准

运行一条命令即可执行测试；README 能让别人从空环境启动项目；你能解释训练集/验证集/测试集的区别，以及为什么不能用测试集调参。

## 建议的第一批命令

```powershell
python --version
python -c "import numpy, pandas, sklearn, pytest; print(numpy.__version__, pandas.__version__, sklearn.__version__, pytest.__version__)"
pytest
ruff check .
```

不需要为了本路线重复创建虚拟环境；只要当前环境能够稳定复现即可。后续若依赖冲突，再单独锁定项目环境。
