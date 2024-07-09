# 05. PyTorch 模块化

[05. PyTorch 模块化](https://www.learnpytorch.io/05_pytorch_going_modular/)部分的主要目标是：**将有用的笔记本代码单元转换为可重用的 Python 脚本（`.py` 文件）**。

本目录包含实现这一目标所需的所有材料。

具体内容如下：
* `going_modular/` - 用于运行 PyTorch 代码的 Python 辅助脚本目录（由 `05_pytorch_going_modular_script_mode.ipynb` 生成）。
* `models/` - 运行笔记本 05. 模块化第 1 部分和第 2 部分后得到的训练好的 PyTorch 模型。
* [`05_pytorch_going_modular_cell_mode.ipynb`](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/going_modular/05_pytorch_going_modular_cell_mode.ipynb) - 第 1/2 部分笔记本，用于教授第 05 部分的材料。该笔记本从第 04 部分笔记本中提取最有用的代码并进行简化。
* [`05_pytorch_going_modular_script_mode.ipynb`](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/going_modular/05_pytorch_going_modular_script_mode.ipynb) - 第 2/2 部分笔记本，用于教授第 05 部分的材料。该笔记本将第 1 部分中最有用的代码单元转换为 `going_modular/` 中的 Python 脚本。

在本部分中，我们将看到第 1 部分笔记本（单元模式）如何转换为第 2 部分笔记本（脚本模式）。

这样做将使我们得到一个结构与上述 `going_modular/` 目录相同的目录。
