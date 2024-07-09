<a href="https://colab.research.google.com/github/19z/pytorch-deep-learning/blob/main/00_pytorch_fundamentals.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开"/></a> 

[查看源代码](https://github.com/19z/pytorch-deep-learning/blob/main/00_pytorch_fundamentals.ipynb) | [查看幻灯片](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/slides/00_pytorch_and_deep_learning_fundamentals.pdf) | [观看视频讲解](https://youtu.be/Z_ikDlimN6A?t=76) 

# 00. PyTorch 基础

## 什么是 PyTorch？

[PyTorch](https://pytorch.org/) 是一个开源的机器学习和深度学习框架。

## PyTorch 可以用来做什么？

PyTorch 允许你使用 Python 代码来操作和处理数据，并编写机器学习算法。

## 为什么使用PyTorch？

机器学习研究人员喜爱使用PyTorch。截至2022年2月，PyTorch在[Papers With Code](https://paperswithcode.com/trends)（一个追踪机器学习研究论文及其附带代码库的网站）上成为[最常用的深度学习框架](https://paperswithcode.com/trends)。

PyTorch还在幕后处理了许多事情，比如GPU加速（使你的代码运行更快）。

因此，你可以专注于操作数据和编写算法，而PyTorch会确保其运行速度。

如果像特斯拉和Meta（Facebook）这样的公司使用它来构建模型，这些模型被部署以支持数百个应用程序、驱动数千辆汽车并向数十亿人传递内容，那么它在开发方面显然也是能力出众的。

## 本模块内容概述

本课程分为不同的部分（笔记本）。

每个笔记本涵盖了 PyTorch 中的重要思想和概念。

后续的笔记本建立在前面笔记本的知识基础上（编号从 00、01、02 开始，一直延续到最后）。

本笔记本涉及机器学习和深度学习的基本构建单元——张量。

具体来说，我们将涵盖以下内容：

| **主题** | **内容** |
| ----- | ----- |
| **张量简介** | 张量是所有机器学习和深度学习的基本构建单元。 |
| **创建张量** | 张量可以表示几乎任何类型的数据（图像、文字、数字表格）。 |
| **从张量中获取信息** | 如果你能把信息放入张量，你也会想把它取出来。 |
| **操作张量** | 机器学习算法（如神经网络）涉及以多种不同方式操作张量，例如加法、乘法、组合。 |
| **处理张量形状** | 机器学习中最常见的问题之一是处理形状不匹配（试图将错误形状的张量与其他张量混合）。 |
| **张量索引** | 如果你曾在 Python 列表或 NumPy 数组上进行索引，张量的操作与之非常相似，只是它们可以有更多的维度。 |
| **混合 PyTorch 张量和 NumPy** | PyTorch 使用张量（[`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html)），NumPy 喜欢数组（[`np.ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html)），有时你会希望混合使用这些。 |
| **可重复性** | 机器学习非常实验性，由于它大量使用随机性来工作，有时你希望这种随机性不那么随机。 |
| **在 GPU 上运行张量** | GPU（图形处理单元）使你的代码运行更快，PyTorch 使得在 GPU 上运行代码变得容易。 |


## 在哪里可以获得帮助？

本课程的所有资料都存放在 [GitHub](https://github.com/mrdbourke/pytorch-deep-learning) 上。

如果你遇到问题，也可以在 [讨论页面](https://github.com/mrdbourke/pytorch-deep-learning/discussions) 上提问。

此外，还有 [PyTorch 开发者论坛](https://discuss.pytorch.org/)，这是一个关于 PyTorch 所有内容的非常有帮助的地方。

## 导入 PyTorch

> **注意：** 在运行本笔记本中的任何代码之前，您应该已经完成了 [PyTorch 安装步骤](https://pytorch.org/get-started/locally/)。
>
> 然而，**如果您在 Google Colab 上运行**，一切应该都能正常工作（Google Colab 自带 PyTorch 和其他库的安装）。

让我们从导入 PyTorch 并检查我们使用的版本开始。

```python
import torch
torch.__version__
```

    '1.13.1'

太好了，看起来我们已经安装了 PyTorch 1.10.0+。

这意味着如果您正在学习这些材料，您将看到大多数内容与 PyTorch 1.10.0+ 兼容，但如果您的版本号远高于此，您可能会注意到一些不一致之处。

如果您遇到任何问题，请在课程的 [GitHub 讨论页面](https://github.com/mrdbourke/pytorch-deep-learning/discussions) 上发帖。

## 张量介绍

现在我们已经导入了 PyTorch，是时候学习张量了。

张量是机器学习的基本构建块。

它们的工作是以数值方式表示数据。

例如，你可以将图像表示为一个形状为 `[3, 224, 224]` 的张量，这意味着 `[颜色通道, 高度, 宽度]`，即图像有 `3` 个颜色通道（红、绿、蓝），高度为 `224` 像素，宽度为 `224` 像素。

![从输入图像到图像的张量表示的示例，图像被分解为 3 个颜色通道以及表示高度和宽度的数字](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/00-tensor-shape-example-of-image.png)

在张量术语（用于描述张量的语言）中，该张量具有三个维度，一个用于 `颜色通道`、`高度` 和 `宽度`。

但我们有点超前了。

让我们通过编码来了解更多关于张量的信息。


### 创建张量

PyTorch 喜欢张量。如此之多以至于有一个完整的文档页面专门介绍 [`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html) 类。

你的第一个作业是 [阅读 `torch.Tensor` 的文档](https://pytorch.org/docs/stable/tensors.html) 10 分钟。但你可以稍后再做。

让我们开始编码。

我们要创建的第一个东西是一个 **标量**。

标量是一个单一的数字，在张量术语中，它是一个零维张量。

> **注意：** 这是本课程的一个趋势。我们将专注于编写特定的代码。但通常我会设置一些涉及阅读和熟悉 PyTorch 文档的练习。因为毕竟，一旦你完成了这门课程，你无疑会想学习更多。而文档是你经常会去的地方。


```python
# 标量
scalar = torch.tensor(7)
scalar
```


    tensor(7)


看看上面的输出是 `tensor(7)` 吗？

这意味着尽管 `scalar` 是一个单独的数字，但它的类型是 `torch.Tensor`。

我们可以使用 `ndim` 属性来检查张量的维度。

```python
scalar.ndim
```


    0



如果我们想要从张量中提取数字呢？

就像，把它从 `torch.Tensor` 转换成一个 Python 整数？

为此，我们可以使用 `item()` 方法。

好的，现在让我们来看一个**向量**。

向量是单维的张量，但可以包含多个数字。

比如，你可以用向量 `[3, 2]` 来描述你家的 `[卧室, 浴室]`。或者你可以用 `[3, 2, 2]` 来描述你家的 `[卧室, 浴室, 停车位]`。

这里重要的趋势是，向量在它能代表的内容上是灵活的（张量也是如此）。

```python
# 向量
vector = torch.tensor([7, 7])
vector
```

太好了，`vector` 现在包含了两个 7，这是我最喜欢的数字。

你觉得它会有多少个维度呢？

```python
# Check the number of dimensions of vector
vector.ndim
```


    1

嗯，这有点奇怪，`vector`包含两个数字，但只有一个维度。

我来告诉你一个小窍门。

你可以通过外部方括号（`[`）的数量来判断PyTorch中张量的维度数量，你只需要数一边。

`vector`有多少个方括号？

张量的另一个重要概念是它们的`shape`属性。形状告诉你它们内部的元素是如何排列的。

让我们来看看`vector`的形状。


```python
# Check shape of vector
vector.shape
```


    torch.Size([2])


上述返回 `torch.Size([2])`，这意味着我们的向量形状为 `[2]`。这是因为我们在方括号内放置了两个元素（`[7, 7]`）。

现在让我们来看一个**矩阵**。

```python
# Matrix
MATRIX = torch.tensor([[7, 8], 
                       [9, 10]])
MATRIX
```


    tensor([[ 7,  8],
            [ 9, 10]])


哇！更多的数字！矩阵就像向量一样灵活，只不过它们多了一个维度。

```python
# Check number of dimensions
MATRIX.ndim
```


    2


`MATRIX` 有两个维度（你数过外面一侧的方括号数量吗？）。

你觉得它会是什么形状？


```python
MATRIX.shape
```



    torch.Size([2, 2])


我们得到输出 `torch.Size([2, 2])`，因为 `MATRIX` 有两层元素，每层有两列。

那么，我们如何创建一个**张量**呢？

```python
# Tensor
TENSOR = torch.tensor([[[1, 2, 3],
                        [3, 6, 9],
                        [2, 4, 5]]])
TENSOR
```




    tensor([[[1, 2, 3],
             [3, 6, 9],
             [2, 4, 5]]])


哇！这个张量看起来真不错。

我要强调的是，张量几乎可以表示任何东西。

我们刚刚创建的这个张量可以是牛排和杏仁黄油店的销售数据（这两样是我最喜欢的食物）。

![一个简单的张量在Google表格中显示星期几、牛排销售和杏仁黄油销售](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/00_simple_tensor.png)

你认为它有多少维度？（提示：使用方括号计数技巧）


```python
# Check number of dimensions for TENSOR
TENSOR.ndim
```


    3


那么它的形状呢？


```python
# Check shape of TENSOR
TENSOR.shape
```


    torch.Size([1, 3, 3])



好的，它输出的是 `torch.Size([1, 3, 3])`。

维度从外到内排列。

这意味着有一个 3x3 的维度。

![不同张量维度的示例](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/00-pytorch-different-tensor-dimensions.png)

> **注意：** 你可能注意到我用小写字母表示 `scalar` 和 `vector`，用大写字母表示 `MATRIX` 和 `TENSOR`。这是有意的。在实际应用中，你通常会看到标量和向量用小写字母表示，如 `y` 或 `a`。而矩阵和张量用大写字母表示，如 `X` 或 `W`。
>
> 你还可能注意到矩阵和张量这两个名称被互换使用。这是常见的做法。因为在 PyTorch 中，你通常处理的是 `torch.Tensor`（因此得名张量），然而，其内部形状和维度将决定它实际上是什么。

让我们总结一下。

| 名称 | 是什么？ | 维度数量 | 通常/示例（小写或大写） |
| ----- | ----- | ----- | ----- |
| **标量** | 一个数字 | 0 | 小写 (`a`) |
| **向量** | 带有方向的数字（例如，带有方向的风速），但也可以包含许多其他数字 | 1 | 小写 (`y`) |
| **矩阵** | 一个二维数字数组 | 2 | 大写 (`Q`) |
| **张量** | 一个 n 维数字数组 | 可以是任意数量，0 维张量是标量，1 维张量是向量 | 大写 (`X`) |

![标量、向量、矩阵和张量及其外观](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/00-scalar-vector-matrix-tensor.png)

### 随机张量

我们已经明确了张量代表某种形式的数据。

而诸如神经网络等机器学习模型则对张量进行操作并从中寻找模式。

但在使用 PyTorch 构建机器学习模型时，你很少会手动创建张量（就像我们一直在做的那样）。

相反，机器学习模型通常从一个包含大量随机数字的张量开始，并通过处理数据来调整这些随机数字，使其更好地表示数据。

本质上：

`从随机数字开始 -> 观察数据 -> 更新随机数字 -> 观察数据 -> 更新随机数字...`

作为数据科学家，你可以定义机器学习模型的初始状态（初始化）、如何观察数据（表示）以及如何更新（优化）其随机数字。

我们稍后会实际操作这些步骤。

现在，让我们看看如何创建一个包含随机数字的张量。

我们可以使用 [`torch.rand()`](https://pytorch.org/docs/stable/generated/torch.rand.html) 并传入 `size` 参数来实现这一点。


```python
# Create a random tensor of size (3, 4)
random_tensor = torch.rand(size=(3, 4))
random_tensor, random_tensor.dtype
```




    (tensor([[0.9900, 0.1882, 0.1744, 0.7445],
             [0.9445, 0.7044, 0.7024, 0.7877],
             [0.0218, 0.7861, 0.9037, 0.9690]]),
     torch.float32)



`torch.rand()` 的灵活性在于我们可以调整 `size` 为任意所需的大小。

例如，假设你想要一个形状为 `[224, 224, 3]`（`[高度, 宽度, 颜色通道]`）的随机张量。

```python
# Create a random tensor of size (224, 224, 3)
random_image_size_tensor = torch.rand(size=(224, 224, 3))
random_image_size_tensor.shape, random_image_size_tensor.ndim
```




    (torch.Size([224, 224, 3]), 3)


### 零和一

有时候，你只是想用零或一来填充张量。

这种情况在掩码操作中很常见（例如，用零掩码某些张量值，让模型知道不要学习它们）。

让我们用 [`torch.zeros()`](https://pytorch.org/docs/stable/generated/torch.zeros.html) 创建一个充满零的张量。

同样，`size` 参数在这里发挥了作用。


```python
# Create a tensor of all zeros
zeros = torch.zeros(size=(3, 4))
zeros, zeros.dtype
```




    (tensor([[0., 0., 0., 0.],
             [0., 0., 0., 0.],
             [0., 0., 0., 0.]]),
     torch.float32)


我们可以采用相同的方法，使用 [`torch.ones()` ](https://pytorch.org/docs/stable/generated/torch.ones.html) 来创建一个全为1的张量。

```python
# Create a tensor of all ones
ones = torch.ones(size=(3, 4))
ones, ones.dtype
```


    (tensor([[1., 1., 1., 1.],
             [1., 1., 1., 1.],
             [1., 1., 1., 1.]]),
     torch.float32)


### 创建一个范围和类似张量

有时候你可能需要一个数字范围，比如1到10或者0到100。

你可以使用 `torch.arange(start, end, step)` 来实现。

其中：
* `start` = 范围的起始值（例如：0）
* `end` = 范围的结束值（例如：10）
* `step` = 每两个值之间的步数（例如：1）

> **注意：** 在Python中，你可以使用 `range()` 来创建一个范围。然而在PyTorch中，`torch.range()` 已被弃用，未来可能会显示错误。


```python
# Use torch.arange(), torch.range() is deprecated 
zero_to_ten_deprecated = torch.range(0, 10) # Note: this may return an error in the future

# Create a range of values 0 to 10
zero_to_ten = torch.arange(start=0, end=10, step=1)
zero_to_ten
```

    /tmp/ipykernel_2411/193451495.py:2: UserWarning: torch.range is deprecated and will be removed in a future release because its behavior is inconsistent with Python's range builtin. Instead, use torch.arange, which produces values in [start, end).
      zero_to_ten_deprecated = torch.range(0, 10) # Note: this may return an error in the future





    tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


有时候，你可能需要一个特定类型的张量，其形状与另一个张量相同。

例如，一个形状与之前张量相同的全零张量。

为此，你可以使用 [`torch.zeros_like(input)`](https://pytorch.org/docs/stable/generated/torch.zeros_like.html) 或 [`torch.ones_like(input)`](https://pytorch.org/docs/1.9.1/generated/torch.ones_like.html)，它们分别返回一个形状与 `input` 相同、填充了零或一的张量。


```python
# Can also create a tensor of zeros similar to another tensor
ten_zeros = torch.zeros_like(input=zero_to_ten) # will have same shape
ten_zeros
```


    tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


### 张量数据类型

在PyTorch中有许多不同的[张量数据类型](https://pytorch.org/docs/stable/tensors.html#data-types)。

有些专为CPU设计，有些则更适合GPU。

了解它们之间的区别需要一些时间。

通常，如果你在任何地方看到`torch.cuda`，这意味着该张量正被用于GPU（因为Nvidia GPU使用名为CUDA的计算工具包）。

最常见的类型（通常也是默认类型）是`torch.float32`或`torch.float`。

这被称为“32位浮点数”。

但也有16位浮点数（`torch.float16`或`torch.half`）和64位浮点数（`torch.float64`或`torch.double`）。

更复杂的是，还有8位、16位、32位和64位的整数。

还有更多类型！

> **注意：** 整数是像`7`这样的平坦圆整数，而浮点数则带有小数点，如`7.0`。

这一切的原因都与**计算中的精度**有关。

精度是指描述一个数字时所使用的细节量。

精度值越高（8、16、32），表示一个数字所需的细节和数据就越多。

这在深度学习和数值计算中很重要，因为你需要进行如此多的运算，计算时所需的细节越多，使用的计算资源也就越多。

因此，低精度数据类型通常计算速度更快，但在评估指标（如准确性）上会牺牲一些性能（计算速度快但准确性较低）。

> **资源：**
  * 查看 [PyTorch 文档中所有可用张量数据类型的列表](https://pytorch.org/docs/stable/tensors.html#data-types)。
  * 阅读 [维基百科页面了解计算中精度的概述](https://en.wikipedia.org/wiki/Precision_(computer_science))。

让我们看看如何创建具有特定数据类型的张量。我们可以使用 `dtype` 参数来实现这一点。


```python
# Default datatype for tensors is float32
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None, # defaults to None, which is torch.float32 or whatever datatype is passed
                               device=None, # defaults to None, which uses the default tensor type
                               requires_grad=False) # if True, operations perfromed on the tensor are recorded 

float_32_tensor.shape, float_32_tensor.dtype, float_32_tensor.device
```




    (torch.Size([3]), torch.float32, device(type='cpu'))



除了形状问题（张量形状不匹配），在PyTorch中你还会遇到另外两种最常见的问题：数据类型和设备问题。

例如，一个张量是 `torch.float32`，而另一个是 `torch.float16`（PyTorch通常希望张量具有相同的格式）。

或者一个张量在CPU上，而另一个在GPU上（PyTorch希望张量之间的计算在同一设备上进行）。

我们稍后会更多地讨论设备问题。

现在，让我们创建一个 `dtype=torch.float16` 的张量。

```python
float_16_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=torch.float16) # torch.half would also work

float_16_tensor.dtype
```




    torch.float16



## 从张量中获取信息

一旦你创建了张量（或者其他人或PyTorch模块为你创建了它们），你可能希望从中获取一些信息。

我们之前已经见过这些，但最常见的三个属性是：
* `shape` - 张量的形状是什么？（某些操作需要特定的形状规则）
* `dtype` - 张量中的元素存储为什么数据类型？
* `device` - 张量存储在哪个设备上？（通常是GPU或CPU）

让我们创建一个随机张量并获取其详细信息。


```python
# Create a tensor
some_tensor = torch.rand(3, 4)

# Find out details about it
print(some_tensor)
print(f"Shape of tensor: {some_tensor.shape}")
print(f"Datatype of tensor: {some_tensor.dtype}")
print(f"Device tensor is stored on: {some_tensor.device}") # will default to CPU
```

    tensor([[0.9270, 0.6217, 0.9093, 0.1493],
            [0.4354, 0.6207, 0.9224, 0.0312],
            [0.3300, 0.0959, 0.6050, 0.7674]])
    Shape of tensor: torch.Size([3, 4])
    Datatype of tensor: torch.float32
    Device tensor is stored on: cpu


> **注意：** 在 PyTorch 中遇到问题时，很可能是与上述三个属性之一有关。因此，当错误消息出现时，给自己唱一首小歌，叫做“什么，什么，哪里”：
  * “*我的张量是什么形状？它们的 datatype 是什么，存储在哪里？什么形状，什么 datatype，哪里哪里哪里*”


## 操作张量（张量运算）

在深度学习中，数据（图像、文本、视频、音频、蛋白质结构等）被表示为张量。

模型通过研究这些张量并对其执行一系列操作（可能是数百万次）来学习，以创建输入数据中模式的表示。

这些操作通常是以下几种运算的精彩组合：
* 加法
* 减法
* 乘法（逐元素）
* 除法
* 矩阵乘法

就是这样。当然还有一些其他的，但这些是神经网络的基本构建块。

以正确的方式堆叠这些构建块，你可以创建最复杂的神经网络（就像乐高积木一样！）。

### 基本操作

让我们从一些基本操作开始，加法（`+`）、减法（`-`）、乘法（`*`）。

它们的工作方式正如你所想的那样。

```python
# 创建一个张量并对其加一个数
tensor = torch.tensor([1, 2, 3])
tensor + 10
```



    tensor([11, 12, 13])



```python
# 乘以10
tensor * 10
```



    tensor([10, 20, 30])




注意上面的张量值并没有变成 `tensor([110, 120, 130])`，这是因为张量内的值不会改变，除非它们被重新赋值。

```python
# 张量不会改变，除非重新赋值
tensor
```



    tensor([1, 2, 3])





让我们减去一个数，这次我们重新赋值 `tensor` 变量。

```python
# 减去并重新赋值
tensor = tensor - 10
tensor
```



    tensor([-9, -8, -7])




```python
# 加并重新赋值
tensor = tensor + 10
tensor
```




    tensor([1, 2, 3])





PyTorch 还有许多内置函数，如 [`torch.mul()`](https://pytorch.org/docs/stable/generated/torch.mul.html#torch.mul)（乘法的简写）和 [`torch.add()`](https://pytorch.org/docs/stable/generated/torch.add.html) 来执行基本操作。

```python
# 也可以使用 torch 函数
torch.multiply(tensor, 10)
```




    tensor([10, 20, 30])




```python
# 原始张量未改变
tensor
```



    tensor([1, 2, 3])




然而，更常见的是使用运算符符号如 `*` 而不是 `torch.mul()`。

```python
# 逐元素乘法（每个元素与其对应元素相乘，索引0->0, 1->1, 2->2）
print(tensor, "*", tensor)
print("等于:", tensor * tensor)
```


    tensor([1, 2, 3]) * tensor([1, 2, 3])
    Equals: tensor([1, 4, 9])



### 矩阵乘法（是你所需要的全部）

机器学习和深度学习算法（如神经网络）中最常见的操作之一是[矩阵乘法](https://www.mathsisfun.com/algebra/matrix-multiplying.html)。

PyTorch 在 [`torch.matmul()`](https://pytorch.org/docs/stable/generated/torch.matmul.html) 方法中实现了矩阵乘法功能。

矩阵乘法要记住的两个主要规则是：
1. **内维度**必须匹配：
  * `(3, 2) @ (3, 2)` 不行
  * `(2, 3) @ (3, 2)` 可以
  * `(3, 2) @ (2, 3)` 可以
2. 结果矩阵具有**外维度**的形状：
 * `(2, 3) @ (3, 2)` -> `(2, 2)`
 * `(3, 2) @ (2, 3)` -> `(3, 3)`

> **注意：** 在 Python 中，`@` 是矩阵乘法的符号。

> **资源：** 你可以在 [PyTorch 文档](https://pytorch.org/docs/stable/generated/torch.matmul.html) 中看到所有矩阵乘法的规则。

让我们创建一个张量并对其执行逐元素乘法和矩阵乘法。

```python
import torch
tensor = torch.tensor([1, 2, 3])
tensor.shape
```



    torch.Size([3])



逐元素乘法和矩阵乘法的区别在于值的相加。

对于我们的 `tensor` 变量，其值为 `[1, 2, 3]`：

| 操作 | 计算 | 代码 |
| ----- | ----- | ----- |
| **逐元素乘法** | `[1*1, 2*2, 3*3]` = `[1, 4, 9]` | `tensor * tensor` |
| **矩阵乘法** | `[1*1 + 2*2 + 3*3]` = `[14]` | `tensor.matmul(tensor)` |

```python
# 逐元素矩阵乘法
tensor * tensor
```


    tensor([1, 4, 9])




```python
# 矩阵乘法
torch.matmul(tensor, tensor)
```



    tensor(14)



```python
# 也可以使用 "@" 符号进行矩阵乘法，但不推荐
tensor @ tensor
```



    tensor(14)



你可以手动进行矩阵乘法，但不推荐。

内置的 `torch.matmul()` 方法更快。

```python
%%time
# 手动矩阵乘法
# （避免使用 for 循环进行操作，它们计算成本高）
value = 0
for i in range(len(tensor)):
  value += tensor[i] * tensor[i]
value
```

    CPU times: user 178 µs, sys: 62 µs, total: 240 µs
    Wall time: 248 µs





    tensor(14)




```python
%%time
torch.matmul(tensor, tensor)
```


    CPU times: user 272 µs, sys: 94 µs, total: 366 µs
    Wall time: 295 µs





    tensor(14)


## 深度学习中最常见的错误之一（形状错误）

由于深度学习很大程度上涉及矩阵的乘法和运算，而矩阵对于形状和大小的组合有严格的规定，因此你在深度学习中最常遇到的错误之一就是形状不匹配。

```python
# Shapes need to be in the right way  
tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]], dtype=torch.float32)

tensor_B = torch.tensor([[7, 10],
                         [8, 11], 
                         [9, 12]], dtype=torch.float32)

torch.matmul(tensor_A, tensor_B) # (this will error)
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    /tmp/ipykernel_1722/2761025649.py in <module>
          8                          [9, 12]], dtype=torch.float32)
          9 
    ---> 10 torch.matmul(tensor_A, tensor_B) # (this will error)
    

    RuntimeError: mat1 and mat2 shapes cannot be multiplied (3x2 and 3x2)


我们可以通过使`tensor_A`和`tensor_B`的内维匹配来实现它们之间的矩阵乘法。

实现这一点的方法之一是使用**转置**（交换给定张量的维度）。

在PyTorch中，你可以使用以下任一方法进行转置：
* `torch.transpose(input, dim0, dim1)` - 其中`input`是要转置的所需张量，`dim0`和`dim1`是要交换的维度。
* `tensor.T` - 其中`tensor`是要转置的所需张量。

让我们尝试后者。

```python
# View tensor_A and tensor_B
print(tensor_A)
print(tensor_B)
```

    tensor([[1., 2.],
            [3., 4.],
            [5., 6.]])
    tensor([[ 7., 10.],
            [ 8., 11.],
            [ 9., 12.]])



```python
# View tensor_A and tensor_B.T
print(tensor_A)
print(tensor_B.T)
```

    tensor([[1., 2.],
            [3., 4.],
            [5., 6.]])
    tensor([[ 7.,  8.,  9.],
            [10., 11., 12.]])



```python
# The operation works when tensor_B is transposed
print(f"Original shapes: tensor_A = {tensor_A.shape}, tensor_B = {tensor_B.shape}\n")
print(f"New shapes: tensor_A = {tensor_A.shape} (same as above), tensor_B.T = {tensor_B.T.shape}\n")
print(f"Multiplying: {tensor_A.shape} * {tensor_B.T.shape} <- inner dimensions match\n")
print("Output:\n")
output = torch.matmul(tensor_A, tensor_B.T)
print(output) 
print(f"\nOutput shape: {output.shape}")
```

    Original shapes: tensor_A = torch.Size([3, 2]), tensor_B = torch.Size([3, 2])
    
    New shapes: tensor_A = torch.Size([3, 2]) (same as above), tensor_B.T = torch.Size([2, 3])
    
    Multiplying: torch.Size([3, 2]) * torch.Size([2, 3]) <- inner dimensions match
    
    Output:
    
    tensor([[ 27.,  30.,  33.],
            [ 61.,  68.,  75.],
            [ 95., 106., 117.]])
    
    Output shape: torch.Size([3, 3])


你也可以使用 [`torch.mm()`](https://pytorch.org/docs/stable/generated/torch.mm.html)，这是 `torch.matmul()` 的简写形式。


```python
# torch.mm is a shortcut for matmul
torch.mm(tensor_A, tensor_B.T)
```




    tensor([[ 27.,  30.,  33.],
            [ 61.,  68.,  75.],
            [ 95., 106., 117.]])



没有转置，矩阵乘法的规则就无法满足，我们会得到如上所示的错误。

来个视觉演示怎么样？

![矩阵乘法的视觉演示](https://github.com/mrdbourke/pytorch-deep-learning/raw/main/images/00-matrix-multiply-crop.gif)

你可以在 http://matrixmultiplication.xyz/ 创建自己的矩阵乘法视觉演示。

> **注意：** 这种矩阵乘法也被称为两个矩阵的[**点积**](https://www.mathsisfun.com/algebra/vectors-dot-product.html)。

神经网络中充满了矩阵乘法和点积。

[`torch.nn.Linear()`](https://pytorch.org/docs/1.9.1/generated/torch.nn.Linear.html) 模块（我们稍后会看到它的实际应用），也称为前馈层或全连接层，实现了输入 `x` 和一个权重矩阵 `A` 之间的矩阵乘法。

$$
y = x\cdot{A^T} + b
$$

解释：
* `x` 是该层的输入（深度学习由多层组成，例如 `torch.nn.Linear()` 等层层叠加）。
* `A` 是该层创建的权重矩阵，初始时为随机数，随着神经网络学习更好地表示数据中的模式而调整（注意 "`T`"，这是因为权重矩阵被转置了）。
  * **注意：** 你也可能经常看到用 `W` 或其他字母如 `X` 来表示权重矩阵。
* `b` 是用于稍微偏移权重和输入的偏置项。
* `y` 是输出（通过对输入进行操作，以期发现其中的模式）。

这是一个线性函数（你可能在高中或其他地方见过类似 $y = mx+b$ 的形式），可以用来绘制一条直线！

让我们来玩转一下线性层。

尝试更改下面的 `in_features` 和 `out_features` 的值，看看会发生什么。

你注意到形状方面有什么变化吗？

```python
# Since the linear layer starts with a random weights matrix, let's make it reproducible (more on this later)
torch.manual_seed(42)
# This uses matrix multiplication
linear = torch.nn.Linear(in_features=2, # in_features = matches inner dimension of input 
                         out_features=6) # out_features = describes outer value 
x = tensor_A
output = linear(x)
print(f"Input shape: {x.shape}\n")
print(f"Output:\n{output}\n\nOutput shape: {output.shape}")
```

    Input shape: torch.Size([3, 2])
    
    Output:
    tensor([[2.2368, 1.2292, 0.4714, 0.3864, 0.1309, 0.9838],
            [4.4919, 2.1970, 0.4469, 0.5285, 0.3401, 2.4777],
            [6.7469, 3.1648, 0.4224, 0.6705, 0.5493, 3.9716]],
           grad_fn=<AddmmBackward0>)
    
    Output shape: torch.Size([3, 6])


> **问题：** 如果将上述代码中的 `in_features` 从 2 改为 3，会发生什么？会报错吗？如何改变输入（`x`）的形状以适应错误？提示：我们之前对 `tensor_B` 做了什么？

如果你以前从未接触过矩阵乘法，一开始可能会感到困惑。

但当你多次尝试并深入研究一些神经网络后，你会发现矩阵乘法无处不在。

记住，矩阵乘法就是你所需要的全部。

![矩阵乘法就是你所需要的全部](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/00_matrix_multiplication_is_all_you_need.jpeg)

*当你开始深入研究神经网络层并构建自己的网络时，你会发现矩阵乘法无处不在。**来源：** https://marksaroufim.substack.com/p/working-class-deep-learner*

### 查找最小值、最大值、均值、总和等（聚合）

现在我们已经了解了几种操作张量的方法，接下来让我们通过几种方法来聚合它们（从更多的值变为更少的值）。

首先，我们将创建一个张量，然后找出它的最大值、最小值、均值和总和。

```python
# Create a tensor
x = torch.arange(0, 100, 10)
x
```




    tensor([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90])


现在让我们进行一些聚合操作。

```python
print(f"Minimum: {x.min()}")
print(f"Maximum: {x.max()}")
# print(f"Mean: {x.mean()}") # this will error
print(f"Mean: {x.type(torch.float32).mean()}") # won't work without float datatype
print(f"Sum: {x.sum()}")
```

    Minimum: 0
    Maximum: 90
    Mean: 45.0
    Sum: 450



> **注意：** 你可能会发现一些方法，如 `torch.mean()` 要求张量必须是 `torch.float32`（最常见的）或其他特定数据类型，否则操作将失败。

你也可以使用 `torch` 方法来完成上述相同的操作。


```python
torch.max(x), torch.min(x), torch.mean(x.type(torch.float32)), torch.sum(x)
```




    (tensor(90), tensor(0), tensor(45.), tensor(450))



### 位置最小/最大值

你也可以分别使用 [`torch.argmax()`](https://pytorch.org/docs/stable/generated/torch.argmax.html) 和 [`torch.argmin()`](https://pytorch.org/docs/stable/generated/torch.argmin.html) 来找到张量中最大值或最小值出现的位置。

这在仅需要最高（或最低）值的位置而非实际值本身时非常有用（我们将在后面的部分中看到这一应用，例如在使用 [softmax 激活函数](https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html) 时）。


```python
# Create a tensor
tensor = torch.arange(10, 100, 10)
print(f"Tensor: {tensor}")

# Returns index of max and min values
print(f"Index where max value occurs: {tensor.argmax()}")
print(f"Index where min value occurs: {tensor.argmin()}")
```

    Tensor: tensor([10, 20, 30, 40, 50, 60, 70, 80, 90])
    Index where max value occurs: 8
    Index where min value occurs: 0


### 改变张量数据类型

如前所述，深度学习操作中常见的问题之一是张量具有不同的数据类型。

如果一个张量是 `torch.float64` 类型，而另一个是 `torch.float32` 类型，你可能会遇到一些错误。

但有一个解决办法。

你可以使用 [`torch.Tensor.type(dtype=None)`](https://pytorch.org/docs/stable/generated/torch.Tensor.type.html) 方法来改变张量的数据类型，其中 `dtype` 参数是你希望使用的数据类型。

首先，我们将创建一个张量并检查其数据类型（默认是 `torch.float32`）。


```python
# Create a tensor and check its datatype
tensor = torch.arange(10., 100., 10.)
tensor.dtype
```




    torch.float32


现在，我们将创建另一个与之前相同的张量，但将其数据类型更改为 `torch.float16`。

```python
# Create a float16 tensor
tensor_float16 = tensor.type(torch.float16)
tensor_float16
```




    tensor([10., 20., 30., 40., 50., 60., 70., 80., 90.], dtype=torch.float16)



我们可以采取类似的方法来创建一个 `torch.int8` 张量。


```python
# Create a int8 tensor
tensor_int8 = tensor.type(torch.int8)
tensor_int8
```




    tensor([10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=torch.int8)



> **注意：** 不同的数据类型一开始可能会让人感到困惑。但可以这样理解，数字越小（例如 32、16、8），计算机存储的值就越不精确。而存储量越小，通常会导致计算速度更快，模型整体更小。基于移动设备的神经网络通常使用 8 位整数进行运算，它们更小、运行更快，但精度不如 32 位浮点数。更多信息，建议阅读关于[计算机科学中的精度](https://en.wikipedia.org/wiki/Precision_(computer_science))的内容。

> **练习：** 到目前为止，我们已经介绍了不少张量方法，但 [`torch.Tensor` 文档](https://pytorch.org/docs/stable/tensors.html)中还有更多内容。建议花 10 分钟时间浏览一下，看看哪些内容吸引了你的注意。点击它们，然后自己动手写代码看看会发生什么。

### 重塑、堆叠、压缩和解压缩

很多时候，你会希望在不实际改变张量内部值的情况下，重塑或改变张量的维度。

为此，一些常用的方法包括：

| 方法 | 一行描述 |
| ----- | ----- |
| [`torch.reshape(input, shape)`](https://pytorch.org/docs/stable/generated/torch.reshape.html#torch.reshape) | 将 `input` 重塑为 `shape`（如果兼容），也可以使用 `torch.Tensor.reshape()`。 |
| [`torch.Tensor.view(shape)`](https://pytorch.org/docs/stable/generated/torch.Tensor.view.html) | 返回一个不同 `shape` 的原始张量视图，但与原始张量共享相同的数据。 |
| [`torch.stack(tensors, dim=0)`](https://pytorch.org/docs/1.9.1/generated/torch.stack.html) | 沿新维度 (`dim`) 连接一系列 `tensors`，所有 `tensors` 必须具有相同的大小。 |
| [`torch.squeeze(input)`](https://pytorch.org/docs/stable/generated/torch.squeeze.html) | 压缩 `input` 以移除所有值为 `1` 的维度。 |
| [`torch.unsqueeze(input, dim)`](https://pytorch.org/docs/1.9.1/generated/torch.unsqueeze.html) | 在 `dim` 处添加一个值为 `1` 的维度后返回 `input`。 |
| [`torch.permute(input, dims)`](https://pytorch.org/docs/stable/generated/torch.permute.html) | 返回原始 `input` 的视图，其维度按 `dims` 重新排列。 |

为什么要使用这些方法？

因为深度学习模型（神经网络）都是以某种方式操作张量的。由于矩阵乘法的规则，如果形状不匹配，就会遇到错误。这些方法有助于确保你的张量的正确元素与其他张量的正确元素混合。

让我们来尝试一下。

首先，我们将创建一个张量。

```python
# Create a tensor
import torch
x = torch.arange(1., 8.)
x, x.shape
```




    (tensor([1., 2., 3., 4., 5., 6., 7.]), torch.Size([7]))



现在让我们通过 `torch.reshape()` 增加一个额外的维度。

```python
# Add an extra dimension
x_reshaped = x.reshape(1, 7)
x_reshaped, x_reshaped.shape
```




    (tensor([[1., 2., 3., 4., 5., 6., 7.]]), torch.Size([1, 7]))





我们也可以使用 `torch.view()` 来改变张量的视图。

```python
# Change view (keeps same data as original but changes view)
# See more: https://stackoverflow.com/a/54507446/7900723
z = x.view(1, 7)
z, z.shape
```




    (tensor([[1., 2., 3., 4., 5., 6., 7.]]), torch.Size([1, 7]))



但请记住，使用 `torch.view()` 改变张量的视图实际上只是创建了同一个张量的新视图。

因此，改变视图也会改变原始张量。

```python
# Changing z changes x
z[:, 0] = 5
z, x
```




    (tensor([[5., 2., 3., 4., 5., 6., 7.]]), tensor([5., 2., 3., 4., 5., 6., 7.]))


如果我们想将新创建的张量在自身上堆叠五次，可以使用 `torch.stack()` 来实现。

```python
# Stack tensors on top of each other
x_stacked = torch.stack([x, x, x, x], dim=0) # try changing dim to dim=1 and see what happens
x_stacked
```




    tensor([[5., 2., 3., 4., 5., 6., 7.],
            [5., 2., 3., 4., 5., 6., 7.],
            [5., 2., 3., 4., 5., 6., 7.],
            [5., 2., 3., 4., 5., 6., 7.]])


如何从张量中移除所有单一维度？

为此，你可以使用 `torch.squeeze()`（我记得这就像是“挤压”张量，使其只保留大于1的维度）。

```python
print(f"Previous tensor: {x_reshaped}")
print(f"Previous shape: {x_reshaped.shape}")

# Remove extra dimension from x_reshaped
x_squeezed = x_reshaped.squeeze()
print(f"\nNew tensor: {x_squeezed}")
print(f"New shape: {x_squeezed.shape}")
```

    Previous tensor: tensor([[5., 2., 3., 4., 5., 6., 7.]])
    Previous shape: torch.Size([1, 7])
    
    New tensor: tensor([5., 2., 3., 4., 5., 6., 7.])
    New shape: torch.Size([7])


要实现与 `torch.squeeze()` 相反的操作，可以使用 `torch.unsqueeze()` 在特定索引处添加一个值为 1 的维度。

```python
print(f"Previous tensor: {x_squeezed}")
print(f"Previous shape: {x_squeezed.shape}")

## Add an extra dimension with unsqueeze
x_unsqueezed = x_squeezed.unsqueeze(dim=0)
print(f"\nNew tensor: {x_unsqueezed}")
print(f"New shape: {x_unsqueezed.shape}")
```

    Previous tensor: tensor([5., 2., 3., 4., 5., 6., 7.])
    Previous shape: torch.Size([7])
    
    New tensor: tensor([[5., 2., 3., 4., 5., 6., 7.]])
    New shape: torch.Size([1, 7])


您还可以通过`torch.permute(input, dims)`重新排列轴值的顺序，其中`input`会转换成具有新`dims`的*视图*。

```python
# Create tensor with specific shape
x_original = torch.rand(size=(224, 224, 3))

# Permute the original tensor to rearrange the axis order
x_permuted = x_original.permute(2, 0, 1) # shifts axis 0->1, 1->2, 2->0

print(f"Previous shape: {x_original.shape}")
print(f"New shape: {x_permuted.shape}")
```

    Previous shape: torch.Size([224, 224, 3])
    New shape: torch.Size([3, 224, 224])


> **注意**：因为置换返回的是一个*视图*（与原始数据共享相同的数据），所以置换后的张量中的值将与原始张量相同。如果你改变了视图中的值，原始张量中的值也会随之改变。

## 索引（从张量中选择数据）

有时候，你可能会想要从张量中选择特定的数据（例如，仅选择第一列或第二行）。

为此，你可以使用索引。

如果你曾经对 Python 列表或 NumPy 数组进行过索引操作，那么使用张量在 PyTorch 中进行索引操作会非常相似。

```python
# Create a tensor 
import torch
x = torch.arange(1, 10).reshape(1, 3, 3)
x, x.shape
```




    (tensor([[[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]]]),
     torch.Size([1, 3, 3]))

索引值的顺序是从外层维度到内层维度（请查看方括号）。

```python
# Let's index bracket by bracket
print(f"First square bracket:\n{x[0]}") 
print(f"Second square bracket: {x[0][0]}") 
print(f"Third square bracket: {x[0][0][0]}")
```

    First square bracket:
    tensor([[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]])
    Second square bracket: tensor([1, 2, 3])
    Third square bracket: 1


你也可以使用 `:` 来指定“这一维度的所有值”，然后使用逗号（`,`）来添加另一个维度。


```python
# Get all values of 0th dimension and the 0 index of 1st dimension
x[:, 0]
```




    tensor([[1, 2, 3]])




```python
# Get all values of 0th & 1st dimensions but only index 1 of 2nd dimension
x[:, :, 1]
```




    tensor([[2, 5, 8]])




```python
# Get all values of the 0 dimension but only the 1 index value of the 1st and 2nd dimension
x[:, 1, 1]
```




    tensor([5])




```python
# Get index 0 of 0th and 1st dimension and all values of 2nd dimension 
x[0, 0, :] # same as x[0][0]
```




    tensor([1, 2, 3])



索引一开始可能会让人感到相当困惑，尤其是对于较大的张量（我仍然需要多次尝试才能正确地进行索引）。但是，通过一些练习并遵循数据探索者的座右铭（***可视化、可视化、可视化***），你会开始掌握它的窍门。

## PyTorch 张量与 NumPy

由于 NumPy 是一个流行的 Python 数值计算库，PyTorch 具备与它良好交互的功能。

你主要会用到的从 NumPy 到 PyTorch（以及反过来）的两种方法有：
* [`torch.from_numpy(ndarray)`](https://pytorch.org/docs/stable/generated/torch.from_numpy.html) - NumPy 数组 -> PyTorch 张量。
* [`torch.Tensor.numpy()`](https://pytorch.org/docs/stable/generated/torch.Tensor.numpy.html) - PyTorch 张量 -> NumPy 数组。

我们来试试这些方法。


```python
# NumPy array to tensor
import torch
import numpy as np
array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array)
array, tensor
```




    (array([1., 2., 3., 4., 5., 6., 7.]),
     tensor([1., 2., 3., 4., 5., 6., 7.], dtype=torch.float64))



> **注意：** 默认情况下，NumPy 数组创建时数据类型为 `float64`，如果你将其转换为 PyTorch 张量，它会保持相同的数据类型（如上所述）。
>
> 然而，许多 PyTorch 计算默认使用 `float32`。
>
> 因此，如果你想将你的 NumPy 数组（float64）转换为 PyTorch 张量（float64），然后再转换为 PyTorch 张量（float32），你可以使用 `tensor = torch.from_numpy(array).type(torch.float32)`。


因为我们之前重新赋值了 `tensor`，所以如果你改变这个张量，数组将保持不变。


```python
# Change the array, keep the tensor
array = array + 1
array, tensor
```




    (array([2., 3., 4., 5., 6., 7., 8.]),
     tensor([1., 2., 3., 4., 5., 6., 7.], dtype=torch.float64))



如果你想从 PyTorch 张量转换为 NumPy 数组，你可以调用 `tensor.numpy()`。


```python
# Tensor to NumPy array
tensor = torch.ones(7) # create a tensor of ones with dtype=float32
numpy_tensor = tensor.numpy() # will be dtype=float32 unless changed
tensor, numpy_tensor
```




    (tensor([1., 1., 1., 1., 1., 1., 1.]),
     array([1., 1., 1., 1., 1., 1., 1.], dtype=float32))



同样，根据上述规则，如果你改变了原始的 `tensor`，新的 `numpy_tensor` 将保持不变。


```python
# Change the tensor, keep the array the same
tensor = tensor + 1
tensor, numpy_tensor
```




    (tensor([2., 2., 2., 2., 2., 2., 2.]),
     array([1., 1., 1., 1., 1., 1., 1.], dtype=float32))


## 可重复性（试图从随机中剔除随机性）

随着你对神经网络和机器学习的了解加深，你会发现随机性在其中扮演了多么重要的角色。

好吧，这里指的是伪随机性。毕竟，从设计角度来看，计算机本质上是非随机的（每一步都是可预测的），所以它们创造的随机性是模拟出来的（尽管对此也有争议，但既然我不是计算机科学家，就让你自己去探索更多吧）。

那么，这和神经网络以及深度学习有什么关系呢？

我们讨论过，神经网络从随机数开始，用于描述数据中的模式（这些随机数是糟糕的描述），并尝试使用张量运算（以及我们尚未讨论的其他一些方法）来改进这些随机数，以更好地描述数据中的模式。

简而言之：

``从随机数开始 -> 张量运算 -> 尝试变得更好（一次又一次）``

虽然随机性既美妙又强大，但有时候你会希望随机性少一些。

为什么？

这样你才能进行可重复的实验。

例如，你创建了一个能够达到X性能的算法。

然后你的朋友尝试验证你不是疯了。

他们怎么能做到这一点呢？

这就是**可重复性**的作用。

换句话说，你能否在我的电脑上运行相同的代码，得到与你电脑上相同（或非常相似）的结果？

让我们来看一个PyTorch中可重复性的简短示例。

我们将首先创建两个随机张量，既然它们是随机的，你可能会认为它们是不同的，对吧？


```python
import torch

# Create two random tensors
random_tensor_A = torch.rand(3, 4)
random_tensor_B = torch.rand(3, 4)

print(f"Tensor A:\n{random_tensor_A}\n")
print(f"Tensor B:\n{random_tensor_B}\n")
print(f"Does Tensor A equal Tensor B? (anywhere)")
random_tensor_A == random_tensor_B
```

    Tensor A:
    tensor([[0.8016, 0.3649, 0.6286, 0.9663],
            [0.7687, 0.4566, 0.5745, 0.9200],
            [0.3230, 0.8613, 0.0919, 0.3102]])
    
    Tensor B:
    tensor([[0.9536, 0.6002, 0.0351, 0.6826],
            [0.3743, 0.5220, 0.1336, 0.9666],
            [0.9754, 0.8474, 0.8988, 0.1105]])
    
    Does Tensor A equal Tensor B? (anywhere)





    tensor([[False, False, False, False],
            [False, False, False, False],
            [False, False, False, False]])


正如你可能预料的那样，张量的值是不同的。

但如果你想创建两个**相同**值的随机张量呢？

也就是说，张量仍然包含随机值，但它们具有相同的“风味”。

这时 [`torch.manual_seed(seed)`](https://pytorch.org/docs/stable/generated/torch.manual_seed.html) 就派上用场了，其中 `seed` 是一个整数（比如 `42`，但它可以是任何值），用于给随机性添加“风味”。

让我们通过创建一些更具“风味”的随机张量来尝试一下。

```python
import torch
import random

# # Set the random seed
RANDOM_SEED=42 # try changing this to different values and see what happens to the numbers below
torch.manual_seed(seed=RANDOM_SEED) 
random_tensor_C = torch.rand(3, 4)

# Have to reset the seed every time a new rand() is called 
# Without this, tensor_D would be different to tensor_C 
torch.random.manual_seed(seed=RANDOM_SEED) # try commenting this line out and seeing what happens
random_tensor_D = torch.rand(3, 4)

print(f"Tensor C:\n{random_tensor_C}\n")
print(f"Tensor D:\n{random_tensor_D}\n")
print(f"Does Tensor C equal Tensor D? (anywhere)")
random_tensor_C == random_tensor_D
```

    Tensor C:
    tensor([[0.8823, 0.9150, 0.3829, 0.9593],
            [0.3904, 0.6009, 0.2566, 0.7936],
            [0.9408, 0.1332, 0.9346, 0.5936]])
    
    Tensor D:
    tensor([[0.8823, 0.9150, 0.3829, 0.9593],
            [0.3904, 0.6009, 0.2566, 0.7936],
            [0.9408, 0.1332, 0.9346, 0.5936]])
    
    Does Tensor C equal Tensor D? (anywhere)





    tensor([[True, True, True, True],
            [True, True, True, True],
            [True, True, True, True]])


不错！

看起来设置种子起作用了。

> **资源：** 我们刚刚涉及的只是 PyTorch 中可重复性的皮毛。关于一般的可重复性和随机种子，我建议查看：
> * [PyTorch 可重复性文档](https://pytorch.org/docs/stable/notes/randomness.html)（一个好的练习是阅读这份文档 10 分钟，即使你现在不理解它，了解它也很重要）。
> * [维基百科随机种子页面](https://en.wikipedia.org/wiki/Random_seed)（这将提供随机种子和伪随机性的一般概述）。



## 在GPU上运行张量（并进行更快的计算）

深度学习算法需要大量的数值运算。

默认情况下，这些运算通常在CPU（中央处理单元）上进行。

然而，还有另一种常见的硬件称为GPU（图形处理单元），它通常在执行神经网络所需特定类型的运算（矩阵乘法）方面比CPU快得多。

你的计算机可能就有一块GPU。

如果是这样，你应该尽可能地利用它来训练神经网络，因为它很可能会显著加快训练时间。

有几种方法可以首先访问GPU，然后让PyTorch使用GPU。

> **注意：** 在本课程中，当我提到“GPU”时，我指的是启用了[Nvidia GPU with CUDA](https://developer.nvidia.com/cuda-gpus)（CUDA是一个计算平台和API，有助于使GPU用于通用计算而不仅仅是图形）的GPU，除非另有说明。

### 1. 获取GPU

你可能已经知道我在说GPU时的意思。但如果没有，有几种方法可以访问GPU。

| **方法** | **设置难度** | **优点** | **缺点** | **如何设置** |
| ----- | ----- | ----- | ----- | ----- |
| Google Colab | 简单 | 免费使用，几乎不需要设置，可以像分享链接一样轻松地与他人分享工作 | 不保存数据输出，计算能力有限，可能会超时 | [遵循Google Colab指南](https://colab.research.google.com/notebooks/gpu.ipynb) |
| 使用自己的GPU | 中等 | 在本地计算机上运行所有内容 | GPU不免费，需要前期成本 | 遵循[PyTorch安装指南](https://pytorch.org/get-started/locally/) |
| 云计算（AWS, GCP, Azure） | 中等至困难 | 前期成本小，几乎无限的计算能力 | 如果持续运行可能会很昂贵，需要一些时间来正确设置 | 遵循[PyTorch安装指南](https://pytorch.org/get-started/cloud-partners/) |

还有更多的GPU使用选项，但上述三种方法目前足够了。

就我个人而言，我使用Google Colab和我自己的个人电脑进行小规模实验（以及创建本课程），并在需要更多计算能力时转向云资源。

> **资源：** 如果你正在考虑购买自己的GPU但不确定选择哪种，[Tim Dettmers有一个很棒的指南](https://timdettmers.com/2020/09/07/which-gpu-for-deep-learning/)。

要检查你是否可以访问Nvidia GPU，可以运行`!nvidia-smi`，其中`!`（也称为bang）表示“在命令行上运行这个”。

```python
!nvidia-smi
```

    /usr/bin/sh: 1: nvidia-smi: not found

如果你没有可访问的Nvidia GPU，上面的命令会输出类似的内容：

```
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running.
```

在这种情况下，返回并遵循安装步骤。

如果你有GPU，上面的命令会输出类似的内容：

```
Wed Jan 19 22:09:08 2022       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 495.46       Driver Version: 460.32.03    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla P100-PCIE...  Off  | 00000000:00:04.0 Off |                    0 |
| N/A   35C    P0    27W / 250W |      0MiB / 16280MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

### 2. 让PyTorch在GPU上运行

一旦你准备好访问GPU，下一步就是让PyTorch使用GPU来存储数据（张量）和进行数据计算（对张量执行操作）。

为此，你可以使用[`torch.cuda`](https://pytorch.org/docs/stable/cuda.html)包。

与其讨论它，不如尝试一下。

你可以使用[`torch.cuda.is_available()`](https://pytorch.org/docs/stable/generated/torch.cuda.is_available.html#torch.cuda.is_available)来检查PyTorch是否可以访问GPU。

```python
# 检查是否有GPU
import torch
torch.cuda.is_available()
```

    False



如果上面的输出是`True`，PyTorch可以看到并使用GPU，如果是`False`，则看不到GPU，在这种情况下，你需要返回并重新进行安装步骤。

现在，假设你希望设置你的代码，使其在CPU或GPU上运行（如果可用）。

这样，如果你或其他人决定运行你的代码，它将无论他们使用什么计算设备都能工作。

让我们创建一个`device`变量来存储可用的设备类型。

```python
# 设置设备类型
device = "cuda" if torch.cuda.is_available() else "cpu"
device
```

    'cpu'



如果上面的输出是`"cuda"`，这意味着我们可以将所有PyTorch代码设置为使用可用的CUDA设备（GPU），如果输出是`"cpu"`，我们的PyTorch代码将坚持使用CPU。

> **注意：** 在PyTorch中，最好编写[**设备不可知代码**](https://pytorch.org/docs/master/notes/cuda.html#device-agnostic-code)。这意味着代码将在CPU（始终可用）或GPU（如果可用）上运行。

如果你想进行更快的计算，可以使用GPU，但如果你想进行*更快*的计算，可以使用多个GPU。

你可以使用[`torch.cuda.device_count()`](https://pytorch.org/docs/stable/generated/torch.cuda.device_count.html#torch.cuda.device_count)来计算PyTorch可以访问的GPU数量。

```python
# 计算设备数量
torch.cuda.device_count()
```

    0



知道PyTorch可以访问的GPU数量很有帮助，以防你希望在一个GPU上运行特定进程，而在另一个GPU上运行另一个进程（PyTorch还有功能可以让你在*所有*GPU上运行进程）。

### 3. 将张量（和模型）放在GPU上

你可以通过调用[`to(device)`](https://pytorch.org/docs/stable/generated/torch.Tensor.to.html)将张量（和模型，我们稍后会看到）放在特定的设备上。其中`device`是你希望张量（或模型）去的目标设备。

为什么要这样做？

GPU提供的数值计算速度比CPU快得多，并且由于我们的**设备不可知代码**（见上文），如果GPU不可用，它将在CPU上运行。

> **注意：** 使用`to(device)`将张量放在GPU上（例如`some_tensor.to(device)`）会返回该张量的副本，例如相同的张量将在CPU和GPU上。要覆盖张量，请重新赋值：
>
> `some_tensor = some_tensor.to(device)`

让我们尝试创建一个张量并将其放在GPU上（如果可用）。

```python
# 创建张量（默认在CPU上）
tensor = torch.tensor([1, 2, 3])

# 张量不在GPU上
print(tensor, tensor.device)

# 将张量移动到GPU（如果可用）
tensor_on_gpu = tensor.to(device)
tensor_on_gpu
```

    tensor([1, 2, 3]) cpu



    tensor([1, 2, 3])



如果你有可用的GPU，上面的代码将输出类似的内容：

```
tensor([1, 2, 3]) cpu
tensor([1, 2, 3], device='cuda:0')
```

注意第二个张量有`device='cuda:0'`，这意味着它存储在第一个可用的GPU上（GPU从0开始索引，如果有两个GPU可用，它们分别是`'cuda:0'`和`'cuda:1'`，依此类推，直到`'cuda:n'`）。

### 4. 将张量移回CPU

如果我们想将张量移回CPU呢？

例如，如果你想与NumPy交互（NumPy不利用GPU），你会想要这样做。

让我们尝试对我们的`tensor_on_gpu`使用[`torch.Tensor.numpy()`](https://pytorch.org/docs/stable/generated/torch.Tensor.numpy.html)方法。

```python
# 如果张量在GPU上，不能将其转换为NumPy（这会出错）
tensor_on_gpu.numpy()
```


    array([1, 2, 3])



相反，要将张量移回CPU并使其可与NumPy一起使用，可以使用[`Tensor.cpu()`](https://pytorch.org/docs/stable/generated/torch.Tensor.cpu.html)。

这将张量复制到CPU内存中，使其可与CPU一起使用。

```python
# 相反，将张量复制回CPU
tensor_back_on_cpu = tensor_on_gpu.cpu().numpy()
tensor_back_on_cpu
```


    array([1, 2, 3])



上面的代码返回GPU张量在CPU内存中的副本，因此原始张量仍然在GPU上。

```python
tensor_on_gpu
```

    tensor([1, 2, 3])



## 练习

所有练习都专注于实践上述代码。

你应该能够通过参考每个部分或遵循所链接的资源来完成它们。

**资源：**

* [练习模板笔记本 00](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/extras/exercises/00_pytorch_fundamentals_exercises.ipynb)。
* [练习示例解决方案笔记本 00](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/extras/solutions/00_pytorch_fundamentals_exercise_solutions.ipynb)（在查看此内容之前尝试练习）。

1. 文档阅读 - 深度学习（以及一般编程学习）的一个很大部分是熟悉你所使用的某个框架的文档。在本课程的其余部分，我们将大量使用 PyTorch 文档。因此，我建议你花 10 分钟阅读以下内容（如果你现在不理解某些内容也没关系，重点还不是完全理解，而是意识）。查看 [`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html#torch-tensor) 和 [`torch.cuda`](https://pytorch.org/docs/master/notes/cuda.html#cuda-semantics) 的文档。
2. 创建一个形状为 `(7, 7)` 的随机张量。
3. 对第 2 步中的张量与另一个形状为 `(1, 7)` 的随机张量进行矩阵乘法（提示：你可能需要转置第二个张量）。
4. 将随机种子设置为 `0` 并重复第 2 和第 3 步的练习。
5. 说到随机种子，我们看到了如何使用 `torch.manual_seed()` 设置它，但是有 GPU 的等效方法吗？（提示：你需要查看 `torch.cuda` 的文档）。如果有，将 GPU 随机种子设置为 `1234`。
6. 创建两个形状为 `(2, 3)` 的随机张量并将它们都发送到 GPU（你需要有 GPU 才能进行此操作）。在创建张量时设置 `torch.manual_seed(1234)`（这不一定是 GPU 随机种子）。
7. 对第 6 步中创建的张量进行矩阵乘法（再次，你可能需要调整其中一个张量的形状）。
8. 找出第 7 步输出中的最大值和最小值。
9. 找出第 7 步输出中的最大和最小索引值。
10. 创建一个形状为 `(1, 1, 1, 10)` 的随机张量，然后创建一个新的张量，移除所有 `1` 维度，得到形状为 `(10)` 的张量。创建时将种子设置为 `7`，并打印出第一个张量及其形状以及第二个张量及其形状。

## 额外课程

* 花 1 小时浏览 [PyTorch 基础教程](https://pytorch.org/tutorials/beginner/basics/intro.html)（我推荐 [快速入门](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html) 和 [张量](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html) 部分）。
* 要了解更多关于张量如何表示数据的信息，请观看此视频：[什么是张量？](https://youtu.be/f5liqUk0ZTw)