# 驾驶员状态检测

[Distracted Driver Detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection)

![](driver.gif)

## AWS

由于此项目要求的计算量较大，建议使用亚马逊 p2.xlarge 云服务器来完成该项目，目前在弗吉尼亚北部有已经配置好了环境的 AMI 可以使用。参考：[在aws上配置深度学习主机 ](https://zhuanlan.zhihu.com/p/25066187)

## 描述

使用深度学习方法检测驾驶员的状态。

* 输入：一张彩色图片
* 输出：十种状态的概率

状态列表：

* c0: 安全驾驶
* c1: 右手打字
* c2: 右手打电话
* c3: 左手打字
* c4: 左手打电话
* c5: 调收音机
* c6: 喝饮料
* c7: 拿后面的东西
* c8: 整理头发和化妆
* c9: 和其他乘客说话

## 数据

此数据集可以从 kaggle 上下载。[Distracted Driver Detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection)

如果你下载有困难，可以点这里:[百度云](http://pan.baidu.com/s/1dFzd0at)

## 建议

建议使用 OpenCV, tensorflow, Keras 完成该项目。其他的工具也可以尝试，比如 caffe, mxnet 等。

* [OpenCV](https://github.com/opencv/opencv)
* [OpenCV python tutorials](http://docs.opencv.org/3.1.0/d6/d00/tutorial_py_root.html)
* [tensorflow](https://github.com/tensorflow/tensorflow)
* [Keras](https://github.com/fchollet/keras)
* [Keras 中文文档](http://keras-cn.readthedocs.io/)

### 建议模型

如果你不知道如何去构建你的模型，可以尝试以下的模型，后面的数字代表年份和月份：

* [LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) 1998
* [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) 12
* [VGGNet](https://arxiv.org/abs/1409.1556) 14.09
* [GoogLeNet](https://arxiv.org/abs/1409.4842) 14.09
* [ResNet](https://arxiv.org/abs/1512.03385) 15.12
* [Inception v3](https://arxiv.org/abs/1512.00567) 15.12
* [Inception v4](https://arxiv.org/abs/1602.07261) 16.02
* [Xception](https://arxiv.org/abs/1610.02357) 16.10
* [ResNeXt](https://arxiv.org/abs/1611.05431) 16.11

参考代码：[deep learning models for keras](https://github.com/fchollet/deep-learning-models)

### 可视化

我们不仅需要模型预测得准，还希望模型能够解释原因，因此我们可以参考这篇论文里的方法，对我们的网络关注的部位可视化。

参考：[Class Activation Mapping](http://cnnlocalization.csail.mit.edu/)

![](cam.jpg)

## 评估

你的项目会由优达学城项目评审师依照[机器学习毕业项目要求](https://review.udacity.com/#!/rubrics/273/view)来评审。请确定你已完整的读过了这个要求，并在提交前对照检查过了你的项目。提交项目必须满足所有要求中每一项才能算作项目通过。

## 提交

* PDF 报告文件
* 数据预处理代码（建议使用 jupyter notebook ）
* 模型训练代码（建议使用 jupyter notebook ）
* notebook 导出的 html 文件
* 包含使用的库，机器硬件，机器操作系统，训练时间等数据的 README 文档（建议使用 Markdown ）
