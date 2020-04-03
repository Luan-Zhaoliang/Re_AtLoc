# -- coding: utf-8 --
# @Time : 01/04/2020 10:31
# @Author : LZL
# @Email: z.luan@qmul.ac.uk
# @File : att.py

import torch
from torch import nn  # torch.nn是专门为神经网络设计的模块化接口。nn构建于autograd之上，可以用来定义和运行神经网络
from torch.nn import functional as F


# Attention模块模型构建
# 在定义自已的网络的时候，需要继承nn.Module类，并重新实现构造函数__init__构造函数和forward这两个方法
class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()  # nn.Module的子类函数必须在构造函数中执行父类的构造函数,等价与nn.Module.__init__()

        self.g = nn.Linear(in_channels, in_channels // 8)  # Attention模型中g参数：输入特征channels,n//8为降采样比例
        self.theta = nn.Linear(in_channels, in_channels // 8)  # Attention模型中theta参数设置
        self.phi = nn.Linear(in_channels, in_channels // 8)  # Attention模型中phi参数

        self.W = nn.Linear(in_channels // 8, in_channels)  # Attention模型中Att(x)中a参数设置

    def forward(self, x):  # x 为输入数据
        batch_size = x.size(0)  # batch size属性设置
        out_channels = x.size(1)  # 输出channels数设置

        g_x = self.g(x).view(batch_size, out_channels // 8, 1)  # g(x)维度变换

        theta_x = self.theta(x).view(batch_size, out_channels // 8, 1)  # theta(x)维度变换
        theta_x = theta_x.permute(0, 2, 1)  # permute 将tensor的维度换位(相当于求转置矩阵)
        phi_x = self.phi(x).view(batch_size, out_channels // 8, 1)  # phi_x维度变换
        f = torch.matmul(phi_x, theta_x)  # matmul表示矩阵a和矩阵b做点乘计算（a和b的维度必须相等）
        f_div_C = F.softmax(f, dim=-1)  # 调用softmax函数（当dim=-1时， 是对某一维度的行进行softmax运算）

        y = torch.matmul(f_div_C, g_x)
        y = y.view(batch_size, out_channels // 8)
        W_y = self.W(y)
        z = W_y + x
        return z
