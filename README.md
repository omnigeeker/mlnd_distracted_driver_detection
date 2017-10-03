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

## 代码说明，依次执行以下步骤：

### 1. 拆分数据集代码

splite_valid.py 

### 2. 基准模型代码

keras-vgg16-visual-finetune.ipynb

### 3. 单模型代码

keras-resnet50-visual-finetune.ipynb

keras-inceptionV3-visual-finetune.ipynb

keras-xception-visual-finetune.ipynb

### 4. 混合模型代码

生成混合模型的输入；write_bottleneck_with_fine_tune.py

最终模型执行代码：main-finetune.ipynb

## 下面是废弃的代码，共参考

不做finetune的 生成混合模型的输入：write_bottleneck.py

不做finetune的 最终混合模型代码：main-without-finetune.ipynb

