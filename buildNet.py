# coding:utf-8
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

class NueraLNet(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        #随机初始化权重
        self.bias = [np.random.randn(1, y) for y in sizes[1:]]
        self.weights = [np.random.randn(x, y) for x, y in zip(sizes[:-1], sizes[1:])]

    def get_result(self, images):
        '''
        用权重矩阵计算未经softmax的预测值
        '''
        result = images
        for b, w in zip(self.bias, self.weights):
            result = sigmoid(np.dot(result, w) + b)
        return result

    def train_net(self, trainimage, trainresult, epoch, rate=1,beta=0.1,decay_rate=0.99, minibatch=1, test_image=None, test_result=None):
        '''
        训练过程
        学习率衰减：
        参数：衰减系数：decay_rate:0.99
        '''
        train_loss,test_loss,test_acc=[],[],[]
        for i in range(epoch):
            temp_train,nums=0,0
            minibatchimage = [trainimage[k:k+minibatch] for k in range(0, len(trainimage), minibatch)]
            minibatchresult = [trainresult[k:k+minibatch] for k in range(0, len(trainimage), minibatch)]
            for image, result in zip(minibatchimage, minibatchresult):
                nums=nums+1
                train_loss0=self.update_net(image, result, rate,beta)
                temp_train=temp_train+train_loss0
                #self.update_net(image, result, rate,beta)
            train_loss.append(temp_train/nums)
            print("第{0}个epoch结束！".format(i+1))
            if test_image and test_result:
                _,test_loss0,acc=self.test_net(test_image, test_result)
                test_loss.append(test_loss0)
                test_acc.append(acc)
            rate=rate*decay_rate

        #绘制loss曲线，测试集准确率曲线，权重矩阵的热力图
        hot_plot(self.weights[0],'W1')
        hot_plot(self.weights[1],'W2')
        hot_plot(self.bias[0],'B1')
        hot_plot(self.bias[1],'B2')
        plot_loss(epoch,train_loss,test_loss)
        plot_acc(epoch,test_acc)
        return test_acc[-1]/len(test_acc)

    def update_net(self, training_image, training_result, rate,beta):
        '''
        用反向传播的求导公式计算梯度，并对权重进行bach级别的梯度下降更新
        l2正则化:
        正则化系数：beta
        梯度反向传播的更新公式： W(i+1)=W(i)−η(∇e0(W)+λW)=(1-ηλ)W(i)-η∇e0(W)
        '''
        batch_b_error = [np.zeros(b.shape) for b in self.bias]
        batch_w_error = [np.zeros(w.shape) for w in self.weights]
        for image, result in zip(training_image, training_result):
            b_error, w_error = self.get_error(image, result)
            batch_b_error = [bbe + be for bbe, be in zip(batch_b_error, b_error)]
            batch_w_error = [bwe + we for bwe, we in zip(batch_w_error, w_error)]
            #print(result,np.argmax(result))
        self.bias = [(1-beta*(rate/len(training_image)))*b - (rate/len(training_image))*bbe for b, bbe in zip(self.bias, batch_b_error)]
        self.weights = [(1-beta*(rate/len(training_image)))*w - (rate/len(training_image))*bwe for w, bwe in zip(self.weights, batch_w_error)]
        loss=self.Cross_Entropy_loss_train(training_image, training_result)
        return loss

    def get_error(self, image, result):
        '''
        用sigmoid的导数公式以及输出计算误差
        '''
        b_error = [np.zeros(b.shape) for b in self.bias]
        w_error = [np.zeros(w.shape) for w in self.weights]
        out_data = [image]
        in_data = []
        for b, w in zip(self.bias, self.weights):
            in_data.append(np.dot(out_data[-1], w) + b)
            out_data.append(sigmoid(in_data[-1]))
        b_error[-1] = sigmoid_prime(in_data[-1]) * (out_data[-1] - result)
        w_error[-1] = np.dot(out_data[-2].transpose(), b_error[-1])
        for l in range(2, self.num_layers):
            b_error[-l] = sigmoid_prime(in_data[-l]) * \
                          np.dot(b_error[-l+1], self.weights[-l+1].transpose())
            w_error[-l] = np.dot(out_data[-l-1].transpose(), b_error[-l])
        return b_error, w_error

    def test_net(self, test_image, test_result):
        '''
        对测试集进行测试
        '''
        results = [(np.argmax(self.get_result(image)), result)
                   for image, result in zip(test_image, test_result)]
        loss=self.Cross_Entropy_loss(test_image,test_result)
        right = sum(int(x == y) for (x, y) in results)
        print("测试集准确率：{0}/{1}".format(right, len(test_result)))
        acc=right/len(test_result)
        return results,loss,acc

    def Cross_Entropy_loss(self,image, result):
        '''
        计算测试集的交叉熵损失函数
        '''
        loss,nums=0,0
        for image, result in zip(image, result):
           loss=loss-math.log(softmax(self.get_result(image))[0][int(result)])
           nums=nums+1
        return loss/nums

    def Cross_Entropy_loss_train(self,image, result):
        '''
        计算训练集的交叉熵损失函数
        '''
        loss,nums=0,0
        for image, result in zip(image, result):
           loss=loss-math.log(softmax(self.get_result(image))[0][int(np.argmax(result))])
           nums=nums+1
        return loss/nums
            
    def save_training(self):
        '''
        保存模型的权重和偏置
        '''
        np.savez('./weights.npz', *self.weights)
        np.savez('./bias.npz', *self.bias)

    def read_training(self):
        '''
        读取模型的权重和偏置
        '''
        length = len(self.sizes) - 1
        file_weights = np.load('./weights.npz')
        file_bias = np.load('./bias.npz')
        self.weights = []
        self.bias = []
        for i in range(length):
            index = "arr_" + str(i)
            self.weights.append(file_weights[index])
            self.bias.append(file_bias[index])


def sigmoid(x):
    '''
    sigmoid激活函数
    '''
    return np.longfloat(1.0 / (1.0 + np.exp(-x)))


def sigmoid_prime(x):
    '''
    sigmoid激活函数的导数
    '''
    return sigmoid(x) * (1 - sigmoid(x))

def softmax(x):
    '''
    softmax激活函数
    '''
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def plot_loss(epochs,train_l,val_l):
    '''
    训练集和测试集的loss绘制
    '''
    co=['red',
        'lightblue']
    data_plot=[train_l,val_l]
    labels=['train_loss','test_loss']

    plt.figure(figsize=(8,6))
    for i in range(2):
        #plt.subplot(6,2,i+1)
        plt.plot([i for i in range(1,epochs+1)],data_plot[i],color=co[i],label=labels[i])
        #plt.title(sales_plot.columns[i+1])
        plt.legend(loc="upper right")
    plt.show()
    # plt.figure(figsize=(8,6))
    # for i in range(3):
    #     #plt.subplot(6,2,i+1)
    #     plt.plot([i for i in range(30,epochs+1)],data_plot[i][30:],color=co[i],label=labels[i])
    #     #plt.title(sales_plot.columns[i+1])
    #     plt.legend(loc="upper right")
    # plt.show()
def plot_acc(epochs,acc):
    '''
    测试集的预测准确率绘制
    '''
    plt.figure(figsize=(8,6))
    plt.plot([i for i in range(1,epochs+1)],acc,color='lightblue',label='test_acc')
    plt.legend(loc="lower right")
    plt.show()

def hot_plot(data,title):
    '''
    权重和偏置矩阵的热力图绘制
    '''
    ax = sns.heatmap(data)
    ax.set_title(title)  # 图标题
    plt.show()