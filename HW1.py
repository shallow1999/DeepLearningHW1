# coding: utf-8
from loadMinist import *
from buildNet import *

train_images = decode_idx3_ubyte(train_images_idx3_ubyte_file)
train_labels = decode_idx1_ubyte(train_labels_idx1_ubyte_file)
test_images = decode_idx3_ubyte(test_images_idx3_ubyte_file)
test_labels = decode_idx1_ubyte(test_labels_idx1_ubyte_file)
trainingimages = [(im / 255).reshape(1, 784) for im in train_images]  # 归一化
traininglabels = [vectorized_result(int(i)) for i in train_labels]
testimages = [(im / 255).reshape(1, 784) for im in test_images]
testlabels = [l for l in test_labels]

#网格搜索参数 学习率i/10000，隐层大小j，正则化系数k
rrate,hhidden,rregular,ttest_acc=[],[],[],[]
for i in [1,10,100]:
    for j in [30,50,70,100]:
        for k in  [0.01,0.001,0.0001]:
            print('\n\n该次训练的参数设置')
            print('学习率: %.6f, 隐层神经元数量: %d， 正则化系数：%.5f' % (i/10000, j, k))
            rrate.append(i)
            hhidden.append(j)
            rregular.append(k)
            net = NueraLNet([28 * 28, j, 10])
            temp=net.train_net(trainingimages, traininglabels, 1, i,k,0.99, 32, testimages, testlabels)
            ttest_acc.append(temp)
#采用最优的参数进行训练
idx=ttest_acc.index(max(ttest_acc))
i,j,k=rrate[idx],hhidden[idx],rregular[idx]
print('\n\n用最优参数进行测试：')
print('学习率: %.6f, 隐层神经元数量: %d， 正则化系数：%.5f' % (i/10000, j, k))

#print(type(traininglabels[0][0][0]))
net = NueraLNet([28 * 28, j, 10])
net.train_net(trainingimages, traininglabels,50, i,k,0.99, 32, testimages, testlabels)
net.save_training()
net.read_training()
print('\n先保存训练好的模型后进行测试：')
net.test_net(testimages, testlabels)
print("end")