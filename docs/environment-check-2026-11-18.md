# 本地环境复核（2026-11-18）

## 已确认

- Python 3.12.3
- pip 25.0
- NumPy、pandas、SciPy、scikit-learn
- pytest
- FastAPI、Uvicorn

## 当前缺少

- matplotlib、seaborn：项目 A 可视化需要，建议下周补齐
- Jupyter：不是必需；优先用 `.py` 脚本保证可复现
- ruff：建议补齐，用于代码质量检查
- PyTorch、Transformers：进入深度学习阶段再安装
- MLflow：实验规模变大后再安装
- LangChain、ChromaDB：RAG 阶段再按实际方案选择，不提前堆依赖

## GPU

当前命令行找不到 `nvidia-smi`，因此暂时按 CPU 学习和开发规划。经典机器学习、NumPy、scikit-learn、FastAPI 和小型 PyTorch 实验都可以在 CPU 上完成；进入深度学习阶段再确认是否需要 CUDA 或云 GPU。

## 结论

当前环境足以立即开始第 1-2 阶段，不需要重装 Python，也不需要一次性安装全部 AI 依赖。建议只补充：

```powershell
python -m pip install matplotlib seaborn ruff
```

如果你希望保持环境完全不变，也可以先只安装 matplotlib，其他工具按阶段安装。
