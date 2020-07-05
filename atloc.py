# -- coding: utf-8 --
# @Time : 01/04/2020 09:44
# @Author : LZL
# @Email: z.luan@qmul.ac.uk
# @File : atloc.py

import torch
import torch.nn as nn  # torch.nn是专门为神经网络设计的模块化接口。nn构建于autograd之上，可以用来定义和运行神经网络
import torch.nn.functional as F
import torch.nn.init  # 提供了常用的初始化方法函数
from network.att import AttentionBlock


# nn.Module：神经网络模块。是一种方便封装参数的方式，具有将参数移动到GPU、导出、加载等功能
# 在定义自已的网络的时候，需要继承nn.Module类，并重新实现构造函数__init__构造函数和forward这两个方法
class AtLoc(nn.Module):
    # 一般把网络中具有可学习参数的层（如全连接层、卷积层等）放在构造函数__init__()中
    def __init__(self, feature_extracter, droprate=0.5, pretrained=True, feat_dim=2048):
        super(AtLoc, self).__init__()  # nn.Module的子类函数必须在构造函数中执行父类的构造函数,等价与nn.Module.__init__()
        self.droprate = droprate

        # 在特征提取器中取代最后一个全连接层
        self.feature_extractor = feature_extracter
        self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)  # AdaptiveAvgPool2d-自适应平均池化函数，输出tensor size为1*1
        fe_out_planes = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)  # feat_dim是特征提取器最后提取的特征维度(2048)

        self.att = AttentionBlock(feat_dim)
        self.fc_xyz = nn.Linear(feat_dim, 3)  # fc_xyz为计算出的position
        self.fc_wpqr = nn.Linear(feat_dim, 3)  # fc_wpqr为计算出的rotation

        # 初始化操作
        if pretrained:
            init_modules = [self.feature_extractor.fc, self.fc_xyz, self.fc_wpqr]  # 初始化modules
        else:
            init_modules = self.modules()

        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)  # kaiming正态分布初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)  # 常数初始化

    def forward(self, x):
        x = self.feature_extractor(x)
        x = F.relu(x)

        x = self.att(x.view(x.size(0), -1))  # reshape x

        if self.droprate > 0:
            x = F.dropout(x, p=self.droprate)  # 模型随机失活

        xyz = self.fc_xyz(x)  # 获取位置参数xyz
        wpqr = self.fc_wpqr(x)  # 获取旋转参数wpqr

        return torch.cat((xyz, wpqr), 1)  # torch.cat是将两个张量（tensor）拼接在一起(根据维度1进行拼接)
