
你正在看到的是 [mrdbourke/pytorch-deep-learning](https://github.com/mrdbourke/pytorch-deep-learning/) 的中文翻译版本。

你可以访问 [这里](https://19z.github.io/pytorch-deep-learning/) 查看翻译后的站点。

# 从零到精通：PyTorch 深度学习训练营

欢迎来到 [从零到精通：PyTorch 深度学习训练营](https://dbourke.link/ZTMPyTorch)，这是互联网上学习 PyTorch 的第二佳场所（第一佳场所是 [PyTorch 官方文档](https://pytorch.org/docs/stable/index.html)）。

* **2023年4月更新：** 新的 [PyTorch 2.0 教程](https://www.learnpytorch.io/pytorch_2_intro/) 已上线！由于 PyTorch 2.0 是一个新增功能且向后兼容的版本，所有之前的课程材料仍将与 PyTorch 2.0 兼容。

<div align="center">
    <a href="https://learnpytorch.io">
        <img src="https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/misc-pytorch-course-launch-cover-white-text-black-background.jpg" width=750 alt="pytorch deep learning by zero to mastery cover photo with different sections of the course">
    </a>
</div>

## 本页内容

* [课程材料/大纲](#course-materialsoutline)
* [关于本课程](#about-this-course)
* [状态](#status)（课程制作进度）
* [日志](#log)（课程材料制作过程的日志）

## 课程材料/大纲

* 📖 **在线书籍版本：** 所有课程材料均可在线阅读，访问 [learnpytorch.io](https://learnpytorch.io) 。
* 🎥 **前五部分在 YouTube 上：** 通过观看 [前 25 小时的内容](https://youtu.be/Z_ikDlimN6A)，一天学会 PyTorch。
* 🔬 **课程重点：** 代码、代码、代码，实验、实验、实验。
* 🏃‍♂️ **教学风格：** [https://sive.rs/kimo](https://sive.rs/kimo)。
* 🤔 **提问：** 查看 [GitHub 讨论页面](https://github.com/mrdbourke/pytorch-deep-learning/discussions) 中的现有问题或提出您自己的问题。


| **章节** | **内容概述** | **练习 & 额外课程** | **幻灯片** |
| ----- | ----- | ----- | ----- |
| [00 - PyTorch 基础](https://www.learnpytorch.io/00_pytorch_fundamentals/) | 深度学习和神经网络中使用的许多基本 PyTorch 操作。 | [前往练习 & 额外课程](https://www.learnpytorch.io/00_pytorch_fundamentals/#exercises) | [前往幻灯片](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/slides/00_pytorch_and_deep_learning_fundamentals.pdf) |
| [01 - PyTorch 工作流程](https://www.learnpytorch.io/01_pytorch_workflow/) | 提供解决深度学习问题和使用 PyTorch 构建神经网络的框架。 | [前往练习 & 额外课程](https://www.learnpytorch.io/01_pytorch_workflow/#exercises) | [前往幻灯片](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/slides/01_pytorch_workflow.pdf) |
| [02 - PyTorch 神经网络分类](https://www.learnpytorch.io/02_pytorch_classification/) | 使用 01 章节的 PyTorch 工作流程解决神经网络分类问题。 | [前往练习 & 额外课程](https://www.learnpytorch.io/02_pytorch_classification/#exercises) | [前往幻灯片](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/slides/02_pytorch_classification.pdf) |
| [03 - PyTorch 计算机视觉](https://www.learnpytorch.io/03_pytorch_computer_vision/) | 看看如何使用 01 & 02 章节的工作流程解决计算机视觉问题。 | [前往练习 & 额外课程](https://www.learnpytorch.io/03_pytorch_computer_vision/#exercises) | [前往幻灯片](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/slides/03_pytorch_computer_vision.pdf) |
| [04 - PyTorch 自定义数据集](https://www.learnpytorch.io/04_pytorch_custom_datasets/) | 如何将自定义数据集加载到 PyTorch 中？本章节还将为我们的模块化代码（在 05 章节中介绍）奠定基础。 | [前往练习 & 额外课程](https://www.learnpytorch.io/04_pytorch_custom_datasets/#exercises) | [前往幻灯片](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/slides/04_pytorch_custom_datasets.pdf) |
| [05 - PyTorch 模块化](https://www.learnpytorch.io/05_pytorch_going_modular/) | PyTorch 设计为模块化，让我们将创建的内容转换为一系列 Python 脚本（这是您在野外经常看到的 PyTorch 代码）。 | [前往练习 & 额外课程](https://www.learnpytorch.io/05_pytorch_going_modular/#exercises) | [前往幻灯片](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/slides/05_pytorch_going_modular.pdf) |
| [06 - PyTorch 迁移学习](https://www.learnpytorch.io/06_pytorch_transfer_learning/) | 让我们采用一个表现良好的预训练模型并将其调整为我们自己的问题。 | [前往练习 & 额外课程](https://www.learnpytorch.io/06_pytorch_transfer_learning/#exercises) | [前往幻灯片](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/slides/06_pytorch_transfer_learning.pdf) |
| [07 - 里程碑项目 1：PyTorch 实验跟踪](https://www.learnpytorch.io/07_pytorch_experiment_tracking/) | 我们已经构建了许多模型...跟踪它们的进展不是很好吗？ | [前往练习 & 额外课程](https://www.learnpytorch.io/07_pytorch_experiment_tracking/#exercises) | [前往幻灯片](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/slides/07_pytorch_experiment_tracking.pdf) |
| [08 - 里程碑项目 2：PyTorch 论文复现](https://www.learnpytorch.io/08_pytorch_paper_replicating/) | PyTorch 是机器学习研究中最受欢迎的深度学习框架，让我们通过复现一篇机器学习论文来看看为什么。 | [前往练习 & 额外课程](https://www.learnpytorch.io/08_pytorch_paper_replicating/#exercises) | [前往幻灯片](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/slides/08_pytorch_paper_replicating.pdf) |
| [09 - 里程碑项目 3：模型部署](https://www.learnpytorch.io/09_pytorch_model_deployment/) | 我们已经构建了一个可工作的 PyTorch 模型...如何让其他人使用它？提示：将其部署到互联网上。 | [前往练习 & 额外课程](https://www.learnpytorch.io/09_pytorch_model_deployment/#exercises) | [前往幻灯片](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/slides/09_pytorch_model_deployment.pdf) |
| [PyTorch 额外资源](https://www.learnpytorch.io/pytorch_extra_resources/) | 本课程涵盖了大量的 PyTorch 和深度学习，但机器学习领域广阔，在这里您将找到推荐的书籍和资源：PyTorch 和深度学习，ML 工程，NLP（自然语言处理），时间序列数据，如何找到数据集等。 | - | - |
| [PyTorch 速查表](https://www.learnpytorch.io/pytorch_cheatsheet/) | 对 PyTorch 的一些主要功能进行快速概览，并附上课程和 PyTorch 文档中更多资源的链接。 | - | - |
| [PyTorch 三大常见错误](https://www.learnpytorch.io/pytorch_most_common_errors/) | 概述 PyTorch 中最常见的三个错误（形状、设备和数据类型错误），它们是如何发生的以及如何修复它们。 | - | - |
| [快速 PyTorch 2.0 教程](https://www.learnpytorch.io/pytorch_2_intro/) | 对 PyTorch 2.0 的快速介绍，新功能以及如何开始，并附上更多学习资源的链接。 | - | - |

## 状态

所有材料已完成，视频已发布在Zero to Mastery上！

有关正在进行的工作板，请参见项目页面 - https://github.com/users/mrdbourke/projects/1

* **总视频数：** 321
* **已完成骨架代码：** 00, 01, 02, 03, 04, 05, 06, 07, 08, 09
* **已完成注释（文本）：** 00, 01, 02, 03, 04, 05, 06, 07, 08, 09
* **已完成图片：** 00, 01, 02, 03, 04, 05, 06, 07, 08, 09
* **已完成演示文稿：** 00, 01, 02, 03, 04, 05, 06, 07, 08, 09
* **已完成练习和解答：** 00, 01, 02, 03, 04, 05, 06, 07, 08, 09

有关几乎每日更新，请参见[日志](https://github.com/mrdbourke/pytorch-deep-learning#log)。


## 关于本课程

### 本课程适合谁？

**你：** 是机器学习或深度学习领域的新手，希望学习 PyTorch。

**本课程：** 以实践和代码优先的方式教授你 PyTorch 和许多机器学习概念。

如果你已经有 1 年以上的机器学习经验，本课程可能会有所帮助，但它特别设计为适合初学者。

### 先决条件是什么？

1. 3-6 个月的 Python 编程经验。
2. 至少完成一门机器学习入门课程（尽管这可能可以跳过，我们会提供许多不同主题的资源链接）。
3. 使用过 Jupyter Notebooks 或 Google Colab 的经验（尽管你可以边学边掌握）。
4. 学习的意愿（最重要）。

对于 1 和 2，我推荐 [Zero to Mastery 数据科学与机器学习训练营](https://dbourke.link/ZTMMLcourse)，它将教你机器学习和 Python 的基础知识（我是偏见的，我也教授那门课程）。

### 课程是如何教授的？

所有的课程材料都可以在[learnpytorch.io](https://learnpytorch.io)的在线书籍中免费获取。如果你喜欢阅读，我建议你通过那里的资源进行学习。

如果你更喜欢通过视频学习，课程也采用学徒式教学，即我编写PyTorch代码，你编写PyTorch代码。

课程格言包括“如果有疑问，运行代码”和“实验，实验，再实验！”是有原因的。

我的全部目标就是帮助你做一件事：通过编写PyTorch代码来学习机器学习。

所有的代码都是通过[Google Colab Notebooks](https://colab.research.google.com)（你也可以使用Jupyter Notebooks）编写的，这是一个实验机器学习的绝佳免费资源。

### 完成课程后我会得到什么？

如果你通过视频学习，会有证书和其他奖励。

但证书也就那样。

你可以将这门课程视为机器学习的动力助推器。

到课程结束时，你将编写数百行PyTorch代码。

并且将接触到许多机器学习中最重要的概念。

因此，当你构建自己的机器学习项目或检查使用PyTorch构建的公共机器学习项目时，会感到熟悉；如果不熟悉，至少你知道该去哪里查找。

### 本课程我将构建什么？

我们从 PyTorch 和机器学习的基础知识开始，即使你是机器学习的新手，也能跟上进度。

然后，我们将探索更高级的领域，包括 PyTorch 神经网络分类、PyTorch 工作流程、计算机视觉、自定义数据集、实验跟踪、模型部署，以及我个人最喜欢的：迁移学习，这是一种强大的技术，可以将一个机器学习模型在另一个问题上学习到的知识应用到你自己的问题上！

在此过程中，你将围绕一个名为 FoodVision 的总体项目构建三个里程碑项目，这是一个用于分类食物图像的神经网络计算机视觉模型。

这些里程碑项目将帮助你练习使用 PyTorch 涵盖重要的机器学习概念，并创建一个你可以向雇主展示的作品集，并说“这是我所做的”。

### 如何开始？

你可以在任何设备上阅读材料，但本课程最好在桌面浏览器中查看和编写代码。

本课程使用一个名为 Google Colab 的免费工具。如果你没有使用经验，我建议你先阅读免费的 [Google Colab 入门教程](https://colab.research.google.com/notebooks/basic_features_overview.ipynb)，然后再回来。

开始方法：

1. 点击上面的笔记本或章节链接，例如“[00. PyTorch 基础](https://www.learnpytorch.io/00_pytorch_fundamentals/)”。
2. 点击顶部的“Open in Colab”按钮。
3. 按几次 SHIFT+Enter 并观察发生了什么。


### 我的问题未得到解答

请在[讨论区](https://github.com/mrdbourke/pytorch-deep-learning/discussions)留言或直接给我发邮件：daniel (at) mrdbourke (dot) com。



## Log

Almost daily updates of what's happening.

* 15 May 2023 - PyTorch 2.0 tutorial finished + videos added to ZTM/Udemy, see code: https://www.learnpytorch.io/pytorch_2_intro/
* 13 Apr 2023 - update PyTorch 2.0 notebook
* 30 Mar 2023 - update PyTorch 2.0 notebook with more info/clean code
* 23 Mar 2023 - upgrade PyTorch 2.0 tutorial with annotations and images
* 13 Mar 2023 - add starter code for PyTorch 2.0 tutorial
* 18 Nov 2022 - add a reference for 3 most common errors in PyTorch + links to course sections for more: https://www.learnpytorch.io/pytorch_most_common_errors/
* 9 Nov 2022 - add PyTorch cheatsheet for a very quick overview of the main features of PyTorch + links to course sections: https://www.learnpytorch.io/pytorch_cheatsheet/
* 9 Nov 2022 - full course materials (300+ videos) are now live on Udemy! You can sign up here: https://www.udemy.com/course/pytorch-for-deep-learning/?couponCode=ZTMGOODIES7 (launch deal code valid
  for 3-4 days from this line)
* 4 Nov 2022 - add a notebook for PyTorch Cheatsheet in `extras/` (a simple overview of many of the most important functionality of PyTorch)
* 2 Oct 2022 - all videos for section 08 and 09 published (100+ videos for the last two sections)!
* 30 Aug 2022 - recorded 15 videos for 09, total videos: 321, finished section 09 videos!!!! ... even bigger than 08!!
* 29 Aug 2022 - recorded 16 videos for 09, total videos: 306
* 28 Aug 2022 - recorded 11 videos for 09, total videos: 290
* 27 Aug 2022 - recorded 16 videos for 09, total videos: 279
* 26 Aug 2022 - add finishing touchs to notebook 09, add slides for 09, create solutions and exercises for 09
* 25 Aug 2022 - add annotations and cleanup 09, remove TK's, cleanup images, make slides for 09
* 24 Aug 2022 - add annotations to 09, main takeaways, exercises and extra-curriculum done
* 23 Aug 2022 - add annotations to 09, add plenty of images/slides
* 22 Aug 2022 - add annotations to 09, start working on slides/images
* 20 Aug 2022 - add annotations to 09
* 19 Aug 2022 - add annotations to 09, check out the awesome demos!
* 18 Aug 2022 - add annotations to 09
* 17 Aug 2022 - add annotations to 09
* 16 Aug 2022 - add annotations to 09
* 15 Aug 2022 - add annotations to 09
* 13 Aug 2022 - add annotations to 09
* 12 Aug 2022 - add demo files for notebook 09 to `demos/`, start annotating notebook 09 with explainer text
* 11 Aug 2022 - finish skeleton code for notebook 09, course finishes deploying 2x models, one for FoodVision Mini & one for (secret)
* 10 Aug 2022 - add section for PyTorch Extra Resources (places to learn more about PyTorch/deep learning): https://www.learnpytorch.io/pytorch_extra_resources/
* 09 Aug 2022 - add more skeleton code to notebook 09
* 08 Aug 2022 - create draft notebook for 09, end goal to deploy FoodVision Mini model and make it publically accessible
* 05 Aug 2022 - recorded 11 videos for 08, total videos: 263, section 08 videos finished!... the biggest section so far
* 04 Aug 2022 - recorded 13 videos for 08, total videos: 252
* 03 Aug 2022 - recorded 3 videos for 08, total videos: 239
* 02 Aug 2022 - recorded 12 videos for 08, total videos: 236
* 30 July 2022 - recorded 11 videos for 08, total videos: 224
* 29 July 2022 - add exercises + solutions for 08, see live walkthrough on YouTube: https://youtu.be/tjpW_BY8y3g
* 28 July 2022 - add slides for 08
* 27 July 2022 - cleanup much of 08, start on slides for 08, exercises and extra-curriculum next
* 26 July 2022 - add annotations and images for 08
* 25 July 2022 - add annotations for 08
* 24 July 2022 - launched first half of course (notebooks 00-04) in a single video (25+ hours!!!) on YouTube: https://youtu.be/Z_ikDlimN6A
* 21 July 2022 - add annotations and images for 08
* 20 July 2022 - add annotations and images for 08, getting so close! this is an epic section
* 19 July 2022 - add annotations and images for 08
* 15 July 2022 - add annotations and images for 08
* 14 July 2022 - add annotations for 08
* 12 July 2022 - add annotations for 08, woo woo this is bigggg section!
* 11 July 2022 - add annotations for 08
* 9 July 2022 - add annotations for 08
* 8 July 2022 - add a bunch of annotations to 08
* 6 July 2022 - course launched on ZTM Academy with videos for sections 00-07! 🚀 - https://dbourke.link/ZTMPyTorch
* 1 July 2022 - add annotations and images for 08
* 30 June 2022 - add annotations for 08
* 28 June 2022 - recorded 11 videos for section 07, total video count 213, all videos for section 07 complete!
* 27 June 2022 - recorded 11 videos for section 07, total video count 202
* 25 June 2022 - recreated 7 videos for section 06 to include updated APIs, total video count 191
* 24 June 2022 - recreated 12 videos for section 06 to include updated APIs
* 23 June 2022 - finish annotations for 07, add exercise template and solutions for 07 + video walkthrough on YouTube: https://youtu.be/cO_r2FYcAjU
* 21 June 2022 - make 08 runnable end-to-end, add images and annotations for 07
* 17 June 2022 - fix up 06, 07 v2 for upcoming torchvision version upgrade, add plenty of annotations to 08
* 13 June 2022 - add notebook 08 first version, starting to replicate the Vision Transformer paper
* 10 June 2022 - add annotations for 07 v2
* 09 June 2022 - create 07 v2 for `torchvision` v0.13 (this will replace 07 v1 when `torchvision=0.13` is released)
* 08 June 2022 - adapt 06 v2 for `torchvision` v0.13 (this will replace 06 v1 when `torchvision=0.13` is released)
* 07 June 2022 - create notebook 06 v2 for upcoming `torchvision` v0.13 update (new transfer learning methods)
* 04 June 2022 - add annotations for 07
* 03 June 2022 - huuuuuuge amount of annotations added to 07
* 31 May 2022 - add a bunch of annotations for 07, make code runnable end-to-end
* 30 May 2022 - record 4 videos for 06, finished section 06, onto section 07, total videos 186
* 28 May 2022 - record 10 videos for 06, total videos 182
* 24 May 2022 - add solutions and exercises for 06
* 23 May 2022 - finished annotations and images for 06, time to do exercises and solutions
* 22 May 2202 - add plenty of images to 06
* 18 May 2022 - add plenty of annotations to 06
* 17 May 2022 - added a bunch of annotations for section 06
* 16 May 2022 - recorded 10 videos for section 05, finish videos for section 05 ✅
* 12 May 2022 - added exercises and solutions for 05
* 11 May 2022 - clean up part 1 and part 2 notebooks for 05, make slides for 05, start on exercises and solutions for 05
* 10 May 2022 - huuuuge updates to the 05 section, see the website, it looks pretty: https://www.learnpytorch.io/05_pytorch_going_modular/
* 09 May 2022 - add a bunch of materials for 05, cleanup docs
* 08 May 2022 - add a bunch of materials for 05
* 06 May 2022 - continue making materials for 05
* 05 May 2022 - update section 05 with headings/outline
* 28 Apr 2022 - recorded 13 videos for 04, finished videos for 04, now to make materials for 05
* 27 Apr 2022 - recorded 3 videos for 04
* 26 Apr 2022 - recorded 10 videos for 04
* 25 Apr 2022 - recorded 11 videos for 04
* 24 Apr 2022 - prepared slides for 04
* 23 Apr 2022 - recorded 6 videos for 03, finished videos for 03, now to 04
* 22 Apr 2022 - recorded 5 videos for 03
* 21 Apr 2022 - recorded 9 videos for 03
* 20 Apr 2022 - recorded 3 videos for 03
* 19 Apr 2022 - recorded 11 videos for 03
* 18 Apr 2022 - finish exercises/solutions for 04, added live-coding walkthrough of 04 exercises/solutions on YouTube: https://youtu.be/vsFMF9wqWx0
* 16 Apr 2022 - finish exercises/solutions for 03, added live-coding walkthrough of 03 exercises/solutions on YouTube: https://youtu.be/_PibmqpEyhA
* 14 Apr 2022 - add final images/annotations for 04, begin on exercises/solutions for 03 & 04
* 13 Apr 2022 - add more images/annotations for 04
* 3 Apr 2022 - add more annotations for 04
* 2 Apr 2022 - add more annotations for 04
* 1 Apr 2022 - add more annotations for 04
* 31 Mar 2022 - add more annotations for 04
* 29 Mar 2022 - add more annotations for 04
* 27 Mar 2022 - starting to add annotations for 04
* 26 Mar 2022 - making dataset for 04
* 25 Mar 2022 - make slides for 03
* 24 Mar 2022 - fix error for 03 not working in docs (finally)
* 23 Mar 2022 - add more images for 03
* 22 Mar 2022 - add images for 03
* 20 Mar 2022 - add more annotations for 03
* 18 Mar 2022 - add more annotations for 03
* 17 Mar 2022 - add more annotations for 03
* 16 Mar 2022 - add more annotations for 03
* 15 Mar 2022 - add more annotations for 03
* 14 Mar 2022 - start adding annotations for notebook 03, see the work in progress here: https://www.learnpytorch.io/03_pytorch_computer_vision/
* 12 Mar 2022 - recorded 12 videos for 02, finished section 02, now onto making materials for 03, 04, 05
* 11 Mar 2022 - recorded 9 videos for 02
* 10 Mar 2022 - recorded 10 videos for 02
* 9 Mar 2022 - cleaning up slides/code for 02, getting ready for recording
* 8 Mar 2022 - recorded 9 videos for section 01, finished section 01, now onto 02
* 7 Mar 2022 - recorded 4 videos for section 01
* 6 Mar 2022 - recorded 4 videos for section 01
* 4 Mar 2022 - recorded 10 videos for section 01
* 20 Feb 2022 - recorded 8 videos for section 00, finished section, now onto 01
* 18 Feb 2022 - recorded 13 videos for section 00
* 17 Feb 2022 - recorded 11 videos for section 00
* 16 Feb 2022 - added setup guide
* 12 Feb 2022 - tidy up README with table of course materials, finish images and slides for 01
* 10 Feb 2022 - finished slides and images for 00, notebook is ready for publishing: https://www.learnpytorch.io/00_pytorch_fundamentals/
* 01-07 Feb 2022 - add annotations for 02, finished, still need images, going to work on exercises/solutions today
* 31 Jan 2022 - start adding annotations for 02
* 28 Jan 2022 - add exercies and solutions for 01
* 26 Jan 2022 - lots more annotations to 01, should be finished tomorrow, will do exercises + solutions then too
* 24 Jan 2022 - add a bunch of annotations to 01
* 21 Jan 2022 - start adding annotations for 01
* 20 Jan 2022 - finish annotations for 00 (still need to add images), add exercises and solutions for 00
* 19 Jan 2022 - add more annotations for 00
* 18 Jan 2022 - add more annotations for 00
* 17 Jan 2022 - back from holidays, adding more annotations to 00
* 10 Dec 2021 - start adding annoations for 00
* 9 Dec 2021 - Created a website for the course ([learnpytorch.io](https://learnpytorch.io)) you'll see updates posted there as development continues
* 8 Dec 2021 - Clean up notebook 07, starting to go back through code and add annotations
* 26 Nov 2021 - Finish skeleton code for 07, added four different experiments, need to clean up and make more straightforward
* 25 Nov 2021 - clean code for 06, add skeleton code for 07 (experiment tracking)
* 24 Nov 2021 - Update 04, 05, 06 notebooks for easier digestion and learning, each section should cover a max of 3 big ideas, 05 is now dedicated to turning notebook code into modular code
* 22 Nov 2021 - Update 04 train and test functions to make more straightforward
* 19 Nov 2021 - Added 05 (transfer learning) notebook, update custom data loading code in 04
* 18 Nov 2021 - Updated vision code for 03 and added custom dataset loading code in 04
* 12 Nov 2021 - Added a bunch of skeleton code to notebook 04 for custom dataset loading, next is modelling with custom data
* 10 Nov 2021 - researching best practice for custom datasets for 04
* 9 Nov 2021 - Update 03 skeleton code to finish off building CNN model, onto 04 for loading custom datasets
* 4 Nov 2021 - Add GPU code to 03 + train/test loops + `helper_functions.py`
* 3 Nov 2021 - Add basic start for 03, going to finish by end of week
* 29 Oct 2021 - Tidied up skeleton code for 02, still a few more things to clean/tidy, created 03
* 28 Oct 2021 - Finished skeleton code for 02, going to clean/tidy tomorrow, 03 next week
* 27 Oct 2021 - add a bunch of code for 02, going to finish tomorrow/by end of week
* 26 Oct 2021 - update 00, 01, 02 with outline/code, skeleton code for 00 & 01 done, 02 next
* 23, 24 Oct 2021 - update 00 and 01 notebooks with more outline/code
* 20 Oct 2021 - add v0 outlines for 01 and 02, add rough outline of course to README, this course will focus on less but better
* 19 Oct 2021 - Start repo 🔥, add fundamentals notebook draft v0
