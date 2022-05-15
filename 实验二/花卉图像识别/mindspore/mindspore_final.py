from easydict import EasyDict as edict
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

import mindspore
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
from mindspore.common import dtype as mstype
from mindspore import context
from mindspore.common.initializer import TruncatedNormal
from mindspore import nn
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import Tensor

# # 用于图像数据集增强
# import cv2 as cv
# from AutoAugment_opencv.AA_classification import AutoAugment,RandomAugment

context.set_context(mode=context.GRAPH_MODE, device_target="CPU", device_id=1)

import argparse #创建解析
parser = argparse.ArgumentParser(description="train flower classify", formatter_class=argparse.ArgumentDefaultsHelpFormatter )
parser.add_argument( '--train_url' , type=str , help= 'the path model saved ')
parser .add_argument( '--data_url', type=str,help='the training data') #解析参数
args,unkown = parser.parse_known_args()

path = args.data_url
print(path)
model_path = args.train_url
print(model_path)



# 1.模型运行时的相关变量定义
cfg = edict({
    'data_path': path,
    'data_size':3670,
    'image_width': 100,  # 图片宽度
    'image_height': 100,  # 图片高度
    'batch_size': 32,
    'channel': 3,  # 图片通道数
    'num_class':6,  # 分类类别
    'weight_decay': 0.01,
    'lr':0.0001,  # 学习率
    'dropout_ratio': 0.5,
    'epoch_size': 400,  # 训练次数
    'sigma':0.01,
    
    'save_checkpoint_steps': 1,  # 多少步保存一次模型
    'keep_checkpoint_max': 1,  # 最多保存多少个模型
    'output_directory': model_path,  # 保存模型路径
    'output_prefix': "checkpoint_classification"  # 保存模型文件名字
})

# # 2. 数据集预处理：应用RandomAugment进行数据集扩充，同时减少模型训练过程中的过拟合问题
# # 源数据集-路径
#     #old_dataset_path = 'C:/Users/Charles/Desktop/mindspore-classification/flower_photos_before'
# old_dataset_path = '../flower_photos_before'
# # 扩充及其他防过拟合处理后的数据集-路径
# new_dataset_path = '../flower_photos_processed'

# flower_kinds = os.listdir(old_dataset_path)
# print(flower_kinds)

# RA = RandomAugment(N=2,M=9) # 定义randomAugment随机图像数据增强操作

# # 对6个不同类型的花卉数据集都做1次扩充：主要目的是减少对数据集的过拟合
# for i in range(len(flower_kinds)):
#     folder_path = os.path.join(old_dataset_path,flower_kinds[i])
#     print("Reading Old Folder - 1:")
#     print(folder_path)
#     photo_list = os.listdir(folder_path)
#     processed_img_savePath = os.path.join(new_dataset_path,flower_kinds[i])
#     for j in range(len(photo_list)):
#         img = cv.imread( os.path.join(folder_path,photo_list[i]) )
#         cv.imwrite(os.path.join(processed_img_savePath,photo_list[i]), ra_auged)
#         ra_auged = RA(img)
#         # 处理后图像的文件名
#         processed_img = '%s_processed_v1'%(photo_list[i].split('.')[0]) + '.jpg'
#         cv.imwrite(os.path.join(processed_img_savePath, processed_img), ra_auged)
# # 对数据量较小的数据集，做第2次扩充：主要是为了平衡不同的数据集规模
# flower_kinds.remove('bougainvillea_before')
# flower_kinds.remove('foxglove_before')
# print(flower_kinds)
# for i in range(len(flower_kinds)):
#     folder_path = os.path.join(old_dataset_path,flower_kinds[i])
#     print("Reading Old Folder - 2:")
#     print(folder_path)
#     photo_list = os.listdir(folder_path)
#     processed_img_savePath = os.path.join(new_dataset_path,flower_kinds[i])
#     for j in range(len(photo_list)):
#         img = cv.imread( os.path.join(folder_path,photo_list[i]) )
#         ra_auged = RA(img)
#         # 处理后图像的文件名
#         processed_img = '%s_processed_v2'%(photo_list[i].split('.')[0]) + '.jpg'
#         cv.imwrite(os.path.join(processed_img_savePath, processed_img), ra_auged)


#从目录中读取图像的源数据集。
de_dataset = ds.ImageFolderDataset(cfg.data_path,
                                   class_indexing={'bee_balm':0,'blackberry_lily':1,'blanket_flower':2,'bougainvillea':3,'bromelia':4,'foxglove':5})
#解码前将输入图像裁剪成任意大小和宽高比。
transform_img = CV.RandomCropDecodeResize([cfg.image_width,cfg.image_height], scale=(0.08, 1.0), ratio=(0.75, 1.333))  #改变尺寸
#转换输入图像；形状（H, W, C）为形状（C, H, W）。
hwc2chw_op = CV.HWC2CHW()
#转换为给定MindSpore数据类型的Tensor操作。
type_cast_op = C.TypeCast(mstype.float32)
#将操作中的每个操作应用到此数据集。
de_dataset = de_dataset.map(input_columns="image", num_parallel_workers=8, operations=transform_img)
de_dataset = de_dataset.map(input_columns="image", operations=hwc2chw_op, num_parallel_workers=8)
de_dataset = de_dataset.map(input_columns="image", operations=type_cast_op, num_parallel_workers=8)
de_dataset = de_dataset.shuffle(buffer_size=cfg.data_size)
#划分训练集测试集
(de_train,de_test)=de_dataset.split([0.8,0.2])
#设置每个批处理的行数
#drop_remainder确定是否删除最后一个可能不完整的批（default=False）。
#如果为True，并且如果可用于生成最后一个批的batch_size行小于batch_size行，则这些行将被删除，并且不会传播到子节点。
de_train=de_train.batch(cfg.batch_size, drop_remainder=True)
#重复此数据集计数次数。
de_test=de_test.batch(cfg.batch_size, drop_remainder=True)
print('训练数据集数量：',de_train.get_dataset_size()*cfg.batch_size)#get_dataset_size()获取批处理的大小。
print('测试数据集数量：',de_test.get_dataset_size()*cfg.batch_size)

data_next=de_dataset.create_dict_iterator(output_numpy=True).__next__()
print('通道数/图像长/宽：', data_next['image'].shape)
print('一张图像的标签样式：', data_next['label'])  # 一共5类，用0-4的数字表达类别。

plt.figure()
plt.imshow(data_next['image'][0,...])
plt.colorbar()
plt.grid(False)
plt.show()


# 定义CNN图像识别网络
class Identification_Net(nn.Cell):
    def __init__(self, num_class=5,channel=3,dropout_ratio=0.5,trun_sigma=0.01):  # 默认一共分5类，图片通道数是3
        super(Identification_Net, self).__init__()
        self.num_class = num_class
        self.channel = channel
        self.dropout_ratio = dropout_ratio
        #设置卷积层
        self.conv1 = nn.Conv2d(self.channel, 32,
                               kernel_size=5, stride=1, padding=0,
                               has_bias=True, pad_mode="same",
                               weight_init=TruncatedNormal(sigma=trun_sigma),bias_init='zeros')
        #设置ReLU激活函数
        self.relu = nn.ReLU()
        #设置最大池化层
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2,pad_mode="valid")
        self.conv2 = nn.Conv2d(32, 64,
                               kernel_size=5, stride=1, padding=0,
                               has_bias=True, pad_mode="same",
                               weight_init=TruncatedNormal(sigma=trun_sigma),bias_init='zeros')
        self.conv3 = nn.Conv2d(64, 128,
                               kernel_size=3, stride=1, padding=0,
                               has_bias=True, pad_mode="same",
                               weight_init=TruncatedNormal(sigma=trun_sigma),bias_init='zeros')
        self.conv4 = nn.Conv2d(128, 128,
                               kernel_size=3, stride=1, padding=0,
                               has_bias=True, pad_mode="same",
                               weight_init=TruncatedNormal(sigma=trun_sigma), bias_init='zeros')
        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(6*6*128, 1024,weight_init =TruncatedNormal(sigma=trun_sigma),bias_init = 0.1)
        self.dropout = nn.Dropout(self.dropout_ratio)
        self.fc2 = nn.Dense(1024, 512, weight_init=TruncatedNormal(sigma=trun_sigma), bias_init=0.1)
        self.fc3 = nn.Dense(512, self.num_class, weight_init=TruncatedNormal(sigma=trun_sigma), bias_init=0.1)
    #构建模型
    def construct(self, x):
        x = self.conv1(x)
        #print(x.shape)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv3(x)
        x = self.max_pool2d(x)
        x = self.conv4(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        #print(x.shape)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


net=Identification_Net(num_class=cfg.num_class, channel=cfg.channel, dropout_ratio=cfg.dropout_ratio)
#计算softmax交叉熵。
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
#opt
fc_weight_params = list(filter(lambda x: 'fc' in x.name and 'weight' in x.name, net.trainable_params()))
other_params=list(filter(lambda x: 'fc' not in x.name or 'weight' not in x.name, net.trainable_params()))
group_params = [{'params': fc_weight_params, 'weight_decay': cfg.weight_decay},
                {'params': other_params},
                {'order_params': net.trainable_params()}]
#设置Adam优化器
net_opt = nn.Adam(group_params, learning_rate=cfg.lr, weight_decay=0.0)
#net_opt = nn.Adam(params=net.trainable_params(), learning_rate=cfg.lr, weight_decay=0.1)

model = Model(net, loss_fn=net_loss, optimizer=net_opt, metrics={"acc"})
loss_cb = LossMonitor(per_print_times=de_train.get_dataset_size()*10)
config_ck = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_steps,
                             keep_checkpoint_max=cfg.keep_checkpoint_max)
ckpoint_cb = ModelCheckpoint(prefix=cfg.output_prefix, directory=cfg.output_directory, config=config_ck)
print("============== Starting Training ==============")
model.train(cfg.epoch_size, de_train, callbacks=[loss_cb, ckpoint_cb], dataset_sink_mode=True)

# 使用测试集评估模型，打印总体准确率
metric = model.eval(de_test)
print(metric)

metric = model.eval(de_train)
print(metric)

