# DeepLearningHW1
1.数据集  
首先从字符集中读取出训练集和验证集的图片及其label的数据，以list的形式存储，该部分数据在data.7z的压缩文件中。  
而对于每一个图片则是为numpy数组，对于每一个标签则为浮点数。  
读取出来的是像素的灰度值，在训练前先将其归一化，并将其每一张图片重构为1x784维的数组。  
训练集和测试集的样本量分别为60000和10000。  

2.训练过程  
main文件是HW1.py，包含训练步骤和测试步骤。  
builtNet.py为网络搭建、反向传播与权重更新部分，loadMinist.py为读取数据部分。  
超参数设置：epoch个数为50；采用指数学习率衰减，衰减系数为0.99；batch size大小为32。  
使用网格搜索，对学习率、隐层神经元数量和l2正则化系数进行调参，选择测试集上准确率最高的那组超参数。  
其中激活函数使用的是sigmoid，损失函数使用的是交叉熵损失函数。  
反向传播、梯度计算、带l2正则化的权重更新主要由builtNet.py中定义的get_error()和update_net()两个函数实现；学习率衰减在定义的train_net()函数中实现。  
在网格搜索过程中选择在测试集上准确率最高的超参数作为最优参数。  
用该组最优超参数训练训练集，在过程中画出训练集和测试集的loss图像和测试集的accuracy图像。
用builtNet.py中定义的save_training()函数来保存两个权重矩阵W1（784x50, 50为隐层神经元个数）, W2（50x10）和偏置B1（50x1）, B2（10x1）。  

3.测试过程  
用builtNet.py中定义的函数read_training()读取保存下来的权重和偏置，来进行测试集的测试。  
得到最终测试集的准确率为96.8%。
