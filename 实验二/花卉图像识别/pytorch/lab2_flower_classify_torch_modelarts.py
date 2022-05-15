import glob                # 用于查询符合特定规则的文件路径名
import os                  # 处理文件和目录
import cv2                 # 用于图像处理
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np  # 导入numpy数据库
import matplotlib.pyplot as plt   #导入matplotlib.pyplot模块，主要用于展示图像
from sklearn.model_selection import train_test_split   #从sklearn.model_selection模块导入train_test_split方法，用于拆分数据集

# 所用第三方包的安装方式，安装前更新pip:
# python -m pip install --upgrade pip
# pip install opencv-python # 对应cv2
# pip install tensorflow
# pip install matplotlib
# pip install sklearn

import argparse
# 创建解析
parser = argparse.ArgumentParser(description="train flower classify",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# 添加参数
parser.add_argument('--train_url', type=str,
                    help='the path model saved')
parser.add_argument('--data_url', type=str, help='the training data')
parser.add_argument('--data_test', type=str, help='the test data',default='../test/' )
parser.add_argument('--model_path', type=str, help='the test data',default='./model.pth/' )
# 解析参数
args, unkown = parser.parse_known_args()

path = args.data_url
model_path = args.train_url

#path = './flower_photos/'   # 数据集的相对地址，改为你自己的，建议将数据集放入代码文件夹下

# Todo 对图片进行缩放，统一处理为大小为w*h的图像，具体数据需自己定
w = 224       #设置图片宽度
h = 224       #设置图片高度
c = 3        #设置图片通道为3


def read_img(path):                                                    # 定义函数read_img，用于读取图像数据，并且对图像进行resize格式统一处理
    cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)]  # 创建层级列表cate，用于对数据存放目录下面的数据文件夹进行遍历，os.path.isdir用于判断文件是否是目录，然后对是目录的文件(os.listdir(path))进行遍历
    imgs=[]                                                            # 创建保存图像的空列表
    labels=[]                                                          # 创建用于保存图像标签的空列表
    for idx,folder in enumerate(cate):                                # enumerate函数用于将一个可遍历的数据对象组合为一个索引序列，同时列出数据和下标,一般用在for循环当中
        for im in glob.glob(folder+'/*.jpg'):                         # 利用glob.glob函数搜索每个层级文件下面符合特定格式“/*.jpg”的图片，并进行遍历
            #print('reading the images:%s'%(im))                      # 遍历图像的同时，打印每张图片的“路径+名称”信息
            img=cv2.imread(im)                                        # 利用cv2.imread函数读取每一张被遍历的图像并将其赋值给img
            img=cv2.resize(img,(w,h))                                 # 利用cv2.resize函数对每张img图像进行大小缩放，统一处理为大小为w*h的图像
            imgs.append(img)                                          # 将每张经过处理的图像数据保存在之前创建的imgs空列表当中
            labels.append(idx)                                        # 将每张经过处理的图像的标签数据保存在之前创建的labels列表当中
    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32)   # 利用np.asarray函数对生成的imgs和labels列表数据进行转化，之后转化成数组数据（imgs转成浮点数型，labels转成整数型）


data, label = read_img(path)                                           # 将read_img函数处理之后的数据定义为样本数据data和标签数据label
print("shape of data:",data.shape)                                     # 查看样本数据的大小
print("shape of label:",label.shape)                                   # 查看标签数据的大小


seed = 100           # 设置随机数种子，即seed值
np.random.seed(seed)   # 保证生成的随机数具有可预测性,即相同的种子（seed值）所产生的随机数是相同的

(x_train, x_val, y_train, y_val) = train_test_split(data, label, test_size=0.20, random_state=seed) #拆分数据集一部分为训练集一部分为验证集，拆分比例可调整
x_train = x_train / 255  #训练集图片标准化
x_val = x_val / 255      #测试集图片标准化

flower_dict = {0:'bee',1:'blackberry',2:'blanket',3:'bougainvillea',4:'bromelia',5:'foxglove'} #创建图像标签列表

# 将H*W*C改为C*H*W
x_train = np.transpose(x_train, [0, 3, 1, 2])
x_val = np.transpose(x_val, [0, 3, 1, 2])

# 将数据转换为tensor类型
x_train = torch.tensor(x_train, dtype=torch.float)
x_val = torch.tensor(x_val, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)
# Todo 自行实现模型结构
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 网络层定义
        # 输入[3*128*128]
        ## 第一层卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,  # 输入图片的通道
                out_channels=64,  # 第一层的filter, 数量不能太少. 否则根本学不出来
                kernel_size=3,
                stride=1,
                padding=1
            ),
            # 经过卷积层，输出[64,224,224],传入池化层
            nn.MaxPool2d(kernel_size=2),  # 经过池化，输出[64,112,112]
        )
        ## 第二层卷积
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.MaxPool2d(kernel_size=2),  # 输出[128,56,56]
            # nn.Dropout(p=0.5),
            # nn.ReLU()
        )
        ## 第三层卷积
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.MaxPool2d(kernel_size=2),  # 输出[256,28,28]
        )
        # ## 第四层卷积
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            # 经过卷积层，输出[512,28,28]
            # nn.Dropout(p=0.5),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 输出[512,14,14]
        )
        ## 线性输出层
        # 6种分类，因此out_features为6
        self.linear1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=512 * 14 * 14, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=1000),
            # nn.Linear(in_features=1024, out_features=256),
            nn.ReLU(),
            # nn.Dropout(p=0.3)
        )
        self.output = nn.Linear(in_features=1000, out_features=6)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)  # 保留batch, 将后面的乘到一起
        x = self.linear1(x)
        output = self.output(x)  # 输出[batch,10]
        return output

# Todo 可调整超参数lr，可选择其他优化器
epoches = 40  # 训练轮数
batch_size = 64  # 单次训练量
learning_rate = 0.001
# 将数据和标签封装为DataLoader
Train_DS = TensorDataset(x_train, y_train)
Train_DL = DataLoader(Train_DS, shuffle=False, batch_size=batch_size)

model = CNN()  # 初始化模型
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_function = nn.CrossEntropyLoss()


#编译模型以供训练。使用多分类损失函数'sparse_categorical_crossentropy'，使用metrics=['accuracy']即评估模型在训练和测试时的性能的指标，使用的准确率。
# Todo可选择其他损失函数
def train(model, epoches, batch_size, learning_rate):
    for epoch in range(epoches):
        print("第{}次epoch：".format(epoch))
        for step, (batch_x, batch_y) in enumerate(Train_DL):
            output = model(batch_x)
            loss = loss_function(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 实时显示准确率
            if step % batch_size == 0:
                output_val = model(x_val)
                y_pred = torch.max(output_val, 1)[1].data.numpy()  # max会返回（index,max)
                loss_data = loss.data.numpy()
                accuracy = ((y_pred == y_val.data.numpy()).astype(int).sum()) / float(y_val.size(0))
                print('Epoch: ', epoch, '| train loss: %.4f' % loss_data, '| test accuracy: %.2f' % accuracy)

    return


train(model, epoches, batch_size, learning_rate)



# Todo 可调整超参数
#训练模型，决定训练集和验证集，batch size：进行梯度下降训练模型时每个batch包含的样本数。
#verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
# model.fit(x_train, y_train, epochs=30, validation_data=(x_val, y_val),batch_size=200, verbose=2)
# #输出模型的结构和参数量
# model.summary()
# model.save(model_path)  #保存模型
torch.save({'model': model.state_dict()}, args.train_url+'model.pth')
