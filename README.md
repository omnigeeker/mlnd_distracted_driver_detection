# 驾驶员状态检测

[Distracted Driver Detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection)

![](images/driver.gif)

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

## 报告说明

* 开题报告: proposal.pdf
* 毕业项目报告: capstone.pdf

## 代码说明

* 预处理代码:  拆分数据集 splite_valid.py  预处理代码 write_bottleneck.py

* 模型训练代码：main.ipynb

* 可视化

keras-inceptionV3-visual.ipynb

keras-resnet50-visual.ipynb

keras-vgg16-visual.ipynb

keras-vgg19-visual.ipynb

keras-xception-visual.ipynb
