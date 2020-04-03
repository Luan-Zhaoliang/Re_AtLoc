# -- coding: utf-8 --
# @Time : 31/03/2020 19:37
# @Author : LZL
# @Email: z.luan@qmul.ac.uk
# @File : options.py

import argparse
import os
import torch
from tools import utils


class Options():
    '''新建一个类，用于解析命令行参数和选项的标准模块'''

    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)  # 创建一个解析对象

    def initialize(self):
        # 基本参数设置
        self.parser.add_argument('--data_dir', type=str, default='data')  # 存放所有数据路径
        self.parser.add_argument('--batchsize', type=int, default=64)  # batchsize大小设置
        self.parser.add_argument('--cropsize', type=int, default=256)  # 裁剪图片尺寸设置
        self.parser.add_argument('--print_freq', type=int, default=20)  # 程序结果显示频率
        self.parser.add_argument('--gpus', type=str, default='-1')  # gpu参数设置
        self.parser.add_argument('--nThreads', type=int, default=8, help='threads for loading data')  # 多线程设置
        self.parser.add_argument('--dataset', type=str, default='7Scenes')  # 数据集类别选择
        self.parser.add_argument('--scene', type=str, default='stairs')  # 数据集场景选择
        self.parser.add_argument('--model', type=str, default='AtLoc')  # 选择训练模型名称
        self.parser.add_argument('--seed', type=int, default=7)  # 用于产生相同随机数
        self.parser.add_argument('--logdir', type=str, default='logs')  # 日志文件记录路径
        self.parser.add_argument('--exp_name', type=str, default='name')  # 模型运算结果存储文件夹
        self.parser.add_argument('--skip', type=int, default=10)  # 间隔取数据设置
        self.parser.add_argument('--variable_skip', type=bool, default=False)
        self.parser.add_argument('--real', type=bool, default=False)
        self.parser.add_argument('--steps', type=int, default=3)
        self.parser.add_argument('--val', type=bool, default=True)  # val控制测试集数据操作

        # 训练时参数设置
        self.parser.add_argument('--epochs', type=int, default=100)  # epochs参数设置
        self.parser.add_argument('--beta', type=float, default=-3.0)  # loss函数权重系数beta设置
        self.parser.add_argument('--color_jitter', type=float, default=0.7,
                                 help='0.7 is only for RobotCar, 0.0 for 7Scenes')  # 图像增强参数设置
        self.parser.add_argument('--train_dropout', type=float, default=0.5)  # 训练时随机失活参数设置
        self.parser.add_argument('--val_freq', type=int, default=5)
        self.parser.add_argument('--result_dir', type=str, default='figures')  # 结果图片存储路径
        self.parser.add_argument('--models_dir', type=str, default='models')  # 训练模型参数存储路径
        self.parser.add_argument('--runs_dir', type=str, default='runs')  # 模型运行结果文件夹名称
        self.parser.add_argument('--lr', type=float, default=5e-5)
        self.parser.add_argument('--weight_decay', type=float, default=0.0005)  # 权重衰减率

        # 测试时参数设置
        self.parser.add_argument('--test_dropout', type=float, default=0.0)  # 测试时随机失活参数设置
        self.parser.add_argument('--weights', type=str,
                                 default='trained_model/7Scenes_stairs_AtLoc_False/models/epoch_500.pth.tar')
        self.parser.add_argument('--save_freq', type=int, default=5)  # 运行结果文件存储频率

    def parse(self):
        '''函数parser用于程序解析命令行和参数初始化'''

        self.initialize()  # 调用函数用于命令行解析和参数设置
        self.opt = self.parser.parse_args()  # parse_args()方法进行解析
        str_ids = self.opt.gpus.split(',')  # gpu参数设置
        self.opt.gpus = []
        # 获取有效gpu设备信息
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpus.append(id)

        # 设置有效gpu设备
        if len(self.opt.gpus) > 0:
            torch.cuda.set_device(self.opt.gpus[0])

        args = vars(self.opt)  # vars() 函数返回对象object的属性和属性值的字典对象
        print('-----------Options------------')
        for k, v in sorted(args.items()):
            print('%s:%s' % (str(k), str(v)))
        print('-----------End----------------')

        # 存储各类结果文件
        self.opt.exp_name = '{:s}_{:s}_{:s}'.format(self.opt.dataset, self.opt.scene, self.opt.model)
        expr_dir = os.path.join(self.opt.logdir, self.opt.exp_name)
        self.opt.results_dir = os.path.join(expr_dir, self.opt.models_dir)
        self.opt.models_dir = os.path.join(expr_dir, self.opt.models_dir)
        self.opt.runs_dir = os.path.join(expr_dir, self.opt.runs_dir)
        # 新建模型运行结果文件夹
        utils.madirs([self.opt.logdir, expr_dir, self.opt.runs_dir, self.opt.models_dir, self.opt.results_dir])
        return self.opt
