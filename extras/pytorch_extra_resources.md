# PyTorch 额外资源

尽管《零到精通 PyTorch》课程总时长超过 40 小时，但你很可能在完成课程后仍然充满学习的热情。

毕竟，这门课程是构建 PyTorch 动力的绝佳途径。

以下资源是为了扩展课程内容而收集的。

不过要提醒一下：这里的内容非常丰富。

最好是从每个部分中选择 1 到 2 个资源（或更少）进行深入探索，其余的可以留待以后学习。

哪一个最好呢？

嗯，如果它们能被列入这个清单，你可以认为它们都是优质的资源。

大多数是针对 PyTorch 的，适合作为课程的延伸，但也有少数不是专门针对 PyTorch 的，不过它们在机器学习领域仍然非常有价值。

## 🔥 纯PyTorch资源

- [**PyTorch博客**](https://pytorch.org/blog/) — 从源头上了解PyTorch的最新动态。我大约每个月查看一次博客更新。
- [**PyTorch文档**](https://pytorch.org/docs) — 我们将在课程中多次探索这一点，但仍有许多内容我们未曾涉及。没关系，经常探索并在必要时深入了解。
- [**PyTorch性能调优指南**](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#) — 课程结束后，您可能首先想做的就是让您的PyTorch模型更快（训练和推理），PyTorch性能调优指南将帮助您做到这一点。
- [**PyTorch食谱**](https://pytorch.org/tutorials/recipes/recipes_index.html) — PyTorch食谱是一系列小型教程，展示您可能想要创建的常见PyTorch功能和工作流程，例如[在PyTorch中加载数据](https://pytorch.org/tutorials/recipes/recipes/loading_data_recipe.html)和[在PyTorch中保存和加载用于推理的模型](https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_models_for_inference.html)。
- [**PyTorch生态系统**](https://pytorch.org/ecosystem/) - 一系列基于纯PyTorch构建的工具，为不同领域添加专业功能，从用于3D计算机视觉的[PyTorch3D](https://pytorch3d.org)到用于快速数据增强的[Albumentations](https://github.com/albumentations-team/albumentations)，再到用于模型评估的[TorchMetrics](https://torchmetrics.readthedocs.io/en/stable/)（感谢[Alessandro的提示](https://github.com/mrdbourke/pytorch-deep-learning/issues/64#issuecomment-1175164531)）。
- [**在VSCode中设置PyTorch**](https://code.visualstudio.com/docs/datascience/pytorch-support) — VSCode是最受欢迎的IDE之一。它的PyTorch支持越来越好。在整个Zero to Mastery PyTorch课程中，我们使用Google Colab是因为它的易用性。但很可能您很快就会在VSCode这样的IDE中进行开发。

## 📈 让纯PyTorch更强大/增加功能的库

本课程专注于纯PyTorch（使用最少的外部库），因为如果你知道如何编写纯PyTorch，你就能学会使用各种扩展库。

- [**fast.ai**](https://github.com/fastai/fastai) — fastai是一个开源库，负责处理构建神经网络的许多繁琐部分，并使创建最先进的模型只需几行代码成为可能。他们的免费库、[课程](https://course.fast.ai)和[文档](https://docs.fast.ai)都是世界级的。
- [**MosaicML 提高模型训练效率**](https://github.com/mosaicml/composer) — 训练模型的速度越快，你就能越快地找出有效和无效的方法。MosaicML的开源`Composer`库通过在后台实现加速算法，帮助你用PyTorch更快地训练神经网络，这意味着你可以更快地从现有的PyTorch模型中获得更好的结果。他们的所有代码都是开源的，文档也非常出色。
- [**PyTorch Lightning 减少样板代码**](https://www.pytorchlightning.ai) — PyTorch Lightning负责处理许多在纯PyTorch中经常需要手动完成的步骤，例如编写训练和测试循环、模型检查点、日志记录等。PyTorch Lightning在PyTorch的基础上构建，允许你用更少的代码制作PyTorch模型。

![扩展/增强纯PyTorch的库。](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/extras-001-libraries-to-make-pytorch-better-or-faster.jpeg)

*扩展/增强纯PyTorch的库。*

## 📖 PyTorch 书籍推荐

- [**使用 PyTorch 和 Scikit-Learn 进行机器学习：通过 Sebastian Raschka 编写的 Python 开发机器学习和深度学习模型**](https://www.amazon.com/Machine-Learning-PyTorch-Scikit-Learn-scikit-learn-ebook-dp-B09NW48MR1/dp/B09NW48MR1/) — 一本极佳的机器学习和深度学习入门书籍。从使用 Scikit-Learn 进行传统机器学习算法开始，解决结构化数据（表格或行和列或 Excel 风格）问题，然后切换到如何使用 PyTorch 进行非结构化数据（如计算机视觉和自然语言处理）的深度学习。
- [**Daniel Voigt Godoy 的 PyTorch 逐步系列**](https://pytorchstepbystep.com) — 与 Zero to Mastery PyTorch 课程从代码优先的角度不同，逐步系列从概念优先的角度涵盖 PyTorch 和深度学习，并附有代码示例。该系列有三版，分别是基础、计算机视觉和序列（NLP），是我最喜欢的从零开始学习 PyTorch 的资源之一。
- [**深入深度学习书籍**](https://d2l.ai) — 可能是互联网上最全面的深度学习概念资源，附有 PyTorch、TensorFlow 和 Gluon 的代码示例。而且全部免费！例如，可以查看作者对我们在 [08. PyTorch 论文复现](https://www.learnpytorch.io/08_pytorch_paper_replicating/) 中涉及的 [视觉变换器](https://d2l.ai/chapter_attention-mechanisms-and-transformers/vision-transformer.html) 的解释。
- **额外推荐：** [fast.ai 课程](https://course.fast.ai)（免费在线提供）也有一本免费在线书籍，[使用 fastai 和 PyTorch 进行深度学习](https://course.fast.ai/Resources/book.html)。

![学习 PyTorch 以及深度学习一般知识的教科书。](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/extras-002-books-for-pytorch.jpeg)

*学习 PyTorch 以及深度学习一般知识的教科书。*


## 🏗 机器学习与深度学习工程资源

机器学习工程（也称为 MLOps 或 ML 操作）是将您创建的模型交付给他人的实践。这可能意味着通过公共应用程序或幕后工作来做出商业决策。

以下资源将帮助您了解更多关于部署机器学习模型的步骤。

- **[Chip Huyen 的《设计机器学习系统》](https://www.amazon.com/Designing-Machine-Learning-Systems-Production-Ready/dp/1098107969)** — 如果您想构建一个 ML 系统，了解其他人如何做到这一点会很有帮助。Chip 的书较少关注构建单个机器学习模型（尽管书中有很多关于这方面的内容），而是关注构建一个连贯的 ML 系统。它涵盖了从数据工程到模型构建、模型部署（在线和离线）到模型监控的所有内容。更棒的是，这本书读起来很愉快，可以看出这本书是由一位作家写的（Chip 之前曾写过几本书）。
- **[Goku Mohandas 的 Made With ML](https://madewithml.com)** — 每当我想要学习或参考与 MLOps 相关的内容时，我都会去 [madewithml.com/mlops](https://madewithml.com/#mlops) 看看是否有相关的课程。Made With ML 不仅教你许多不同 ML 模型的基础知识，还介绍了如何构建一个端到端的 ML 系统，并提供了大量的代码和工具示例。
- **[Andriy Burkov 的《机器学习工程》](http://www.mlebook.com)** — 尽管这本书可以在线免费阅读，但我一出版就买了。我多次将其作为参考资料和学习更多关于 ML 工程的内容，它基本上一直放在我的桌上或触手可及的地方。Burkov 很好地抓住了重点，并在必要时引用了进一步的材料。
- **[Full Stack Deep Learning 课程](https://fullstackdeeplearning.com)** — 我第一次参加这个课程是在 2021 年。它不断发展，涵盖了该领域最新的最佳工具。它将教你如何规划一个解决 ML 问题的项目，如何获取或创建数据，如何在 ML 项目出错时进行故障排除，最重要的是，如何构建 ML 驱动的产品。

![提升您机器学习工程技能的资源（围绕构建机器学习模型的所有步骤）。](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/extras-003-places-to-learn-ml-ops.jpeg)

*提升您机器学习工程技能的资源（围绕构建机器学习模型的所有步骤）。*

## 🗃 如何找到数据集

机器学习项目始于数据。

没有数据，就没有机器学习。

以下资源是寻找各种主题和问题领域中开源且通常可直接使用的数据集的最佳选择之一。

- [**Paperswithcode 数据集**](https://paperswithcode.com/datasets) — 搜索最常用和常见的机器学习基准数据集，了解它们包含的内容、来源以及可找到的位置。通常还能看到每个数据集上表现最佳的模型。
- [**HuggingFace 数据集**](https://huggingface.co/docs/datasets) — 不仅是一个跨广泛问题领域查找数据集的资源，还是一个库，可用于几行代码内下载并开始使用这些数据集。
- **[Kaggle 数据集](https://www.kaggle.com/datasets)** — 找到通常伴随 Kaggle 竞赛的各种数据集，其中许多直接来自行业。
- **[Google 数据集搜索](https://datasetsearch.research.google.com)** — 就像使用 Google 搜索一样，但专门针对数据集。

这些资源应该足够开始使用，但对于特定的具体问题，你可能需要构建自己的数据集。

![各种问题领域中现有和开源数据集的寻找地点。](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/extras-004-places-to-find-datasets.jpeg)

*各种问题领域中现有和开源数据集的寻找地点。*


## 深度学习领域的工具

以下资源专注于特定问题领域的库和预训练模型，如计算机视觉和推荐引擎/系统。

### 😎 计算机视觉

我们在 [03. PyTorch 计算机视觉](https://www.learnpytorch.io/03_pytorch_computer_vision/) 中介绍了计算机视觉，但作为快速回顾，计算机视觉是让计算机“看”的艺术。

如果你的数据是视觉的，如图像、X光片、生产线视频甚至手写文档，那么这可能是一个计算机视觉问题。

- **[TorchVision](https://pytorch.org/vision/stable/index.html)** — PyTorch 的计算机视觉库。找到许多加载视觉数据的方法以及许多可用于自己问题的预训练计算机视觉模型。
- [**timm (Torch Image Models) 库**](https://github.com/rwightman/pytorch-image-models) — 最全面的计算机视觉库和预训练计算机视觉模型资源之一。几乎所有使用 PyTorch 进行计算机视觉的新研究都在某种程度上利用了 `timm` 库。
- **[Yolov5 用于目标检测](https://github.com/ultralytics/yolov5)** — 如果你想在 PyTorch 中构建目标检测模型，`yolov5` GitHub 仓库可能是快速入门的最佳方式。
- **[VISSL (Vision Self-Supervised Learning) 库](https://github.com/facebookresearch/vissl)** — 自监督学习是让数据自己学习模式的艺术。与提供不同类别的标签并学习表示不同，自监督学习试图在没有标签的情况下复制类似的结果。VISSL 提供了一种易于使用的方式，通过 PyTorch 开始使用自监督学习计算机视觉模型。

### 📚 自然语言处理 (NLP)

自然语言处理涉及在文本中寻找模式。

例如，你可能想要从支持工单中提取重要实体或将文档分类到不同类别中。

如果你的问题涉及大量文本，你会想要查看以下资源。

- **[TorchText](https://pytorch.org/text/stable/index.html)** — PyTorch 内置的文本领域库。与 TorchVision 类似，它包含许多预构建的方法来加载数据和一系列可适应自己问题的预训练模型。
- [**HuggingFace Transformers 库**](https://huggingface.co/docs/transformers/index) — HuggingFace Transformers 库在 GitHub 上的星数比 PyTorch 库本身还要多。这有其原因。并不是说 HuggingFace Transformers 比 PyTorch 更好，而是因为它在它所做的事情上做得最好：为 NLP（以及更多）提供数据加载器和预训练的最新模型。
- **额外提示：** 想要了解更多关于 HuggingFace Transformers 库及其周边的一切，HuggingFace 团队提供了一个[免费在线课程](https://huggingface.co/course/chapter1/1)。

### 🎤 语音和音频

如果你的问题涉及音频文件或语音数据，如尝试对声音进行分类或将语音转录为文本，你会想要查看以下资源。

- [**TorchAudio**](https://pytorch.org/audio/stable/index.html) — PyTorch 的音频领域库。找到内置的方法来准备数据和预构建的模型架构来寻找音频数据中的模式。
- **[SpeechBrain](https://speechbrain.github.io)** — 一个基于 PyTorch 的开源库，用于处理语音问题，如识别（将语音转为文本）、语音增强、语音处理、文本到语音等。你可以在 [HuggingFace Hub](https://huggingface.co/speechbrain) 上尝试他们的许多模型。

### ❓推荐引擎

互联网是由推荐驱动的。YouTube 推荐视频，Netflix 推荐电影和电视节目，亚马逊推荐产品，Medium 推荐文章。

如果你正在构建一个在线商店或在线市场，你很可能会想开始向你的客户推荐东西。

为此，你会想要构建一个推荐引擎。

- **[TorchRec](https://pytorch.org/torchrec/)** — PyTorch 最新的内置领域库，用于通过深度学习驱动推荐引擎。TorchRec 提供了可以尝试和使用的推荐数据集和模型。尽管如果自定义推荐引擎不符合你的要求（或工作量太大），许多云供应商提供了推荐引擎服务。

### ⏳ 时间序列

如果你的数据有时间组件，并且你希望利用过去的模式来预测未来，例如，预测明年比特币的价格（不要尝试这个，[股票预测是 BS](https://dev.mrdbourke.com/tensorflow-deep-learning/10_time_series_forecasting_in_tensorflow/#model-10-why-forecasting-is-bs-the-turkey-problem)）或更合理的预测下周城市电力需求的问题，你会想要查看时间序列库。

这两个库不一定使用 PyTorch，但由于时间序列是一个常见问题，我在这里包含了它们。

- [**Salesforce Merlion**](https://github.com/salesforce/Merlion) — 通过使用 Merlion 的数据加载器、预构建模型、AutoML（自动化机器学习）超参数调整等，将你的时间序列数据转化为情报，所有这些都是受实际用例启发的，用于时间序列预测和时间序列异常检测。
- [**Facebook Kats**](https://github.com/facebookresearch/Kats) — Facebook 的整个业务依赖于预测：何时是放置广告的最佳时间？所以你可以打赌他们在时间序列预测软件上投入了大量资金。Kats（Kit to Analyze Time Series data）是他们的开源库，用于时间序列预测、检测和数据处理。当然，请提供您希望翻译的英文 Markdown 内容，我会将其翻译为中文，并保持原有格式。


## 👩‍💻 如何找到工作

完成机器学习课程后，你很可能会想运用你的机器学习技能。

甚至更好的是，用它们来赚钱。

以下资源是关于如何找到工作的良好指南。

- **["像我这样的初学者数据科学家如何获得经验？"](https://www.mrdbourke.com/how-can-a-beginner-data-scientist-like-me-gain-experience/) 作者：Daniel Bourke** — 我经常被问到“如何获得经验？”这个问题，因为许多工作要求都写着“需要经验”。事实证明，获得经验（和工作的）最佳方式之一是：*在拥有工作之前就开始做这份工作*。
- **[你并不真的需要另一个MOOC](https://eugeneyan.com/writing/you-dont-need-another-mooc/) 作者：Eugene Yan** — MOOC代表大规模在线公开课程（或类似的东西）。MOOCs非常美好。它们让世界各地的人们按照自己的节奏学习。然而，人们可能会不断重复做MOOC，认为“如果我再做一门，我就会准备好了”。事实上，几门就足够了，MOOC的回报很快就会开始减少。相反，离开常规路径，开始构建，开始创造，开始学习无法被教授的技能。展示这些技能来获得工作。
- **额外推荐：** 关于机器学习面试最全面的资源，请查看Chip Huyen的免费[《机器学习面试指南》](https://huyenchip.com/ml-interviews-book/)。