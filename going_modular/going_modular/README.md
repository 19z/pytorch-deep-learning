# 模块化脚本

本目录中的Python脚本是通过[05. 模块化 Part 2（脚本模式）](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/going_modular/05_pytorch_going_modular_script_mode.ipynb)笔记本生成的。

它们的分工如下：
* `data_setup.py` - 用于准备和下载数据（如果需要）的文件。
* `engine.py` - 包含各种训练函数的文件。
* `model_builder.py` - 用于创建PyTorch TinyVGG模型的文件。
* `train.py` - 利用所有其他文件训练目标PyTorch模型的文件。
* `utils.py` - 专门用于有用的实用函数的文件。
* **额外内容：** `predictions.py` - 使用训练好的PyTorch模型和输入图像进行预测的文件（主要功能`pred_and_plot_image()`最初在[06. PyTorch迁移学习第6节](https://www.learnpytorch.io/06_pytorch_transfer_learning/#6-make-predictions-on-images-from-the-test-set)中创建）。

有关如何实现这一点的解释，请参考[learnpytorch.io书籍的05. PyTorch模块化部分](https://www.learnpytorch.io/05_pytorch_going_modular/)。