# -- coding: utf-8 --
# @Time : 01/04/2020 08:17
# @Author : LZL
# @Email: z.luan@qmul.ac.uk
# @File : train.py

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 保证程序中的GPU序号是和硬件中的序号是相同的
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"  # 限定使用的gpu设备资源

import torch
import sys
import time
import os.path as osp
import numpy as np
import matplotlib

matplotlib.use('TkAgg')  # 设置后端，Matplotlib绘图并保存图像但不显示图形

from tensorboardX import SummaryWriter
from tools.options import Options
from network.atloc import AtLoc
from torchvision import transforms, models
from tools.utils import Logger, AtLocCriterion, AverageMeter
from data.dataloaders import SevenScenes, RobotCar
from torch.utils.data import DataLoader
from torch.autograd import Variable

# 配置运行环境
opt = Options().parse()  # 参数命令解析
cuda = torch.cuda.is_available()  # 判断是否有可用gpu设备
device = "cuda:" + ",".join(str(i) for i in opt.gpus) if cuda else "cpu"  # 获取gpu或cpu设备信息
logfile = osp.join(opt.runs_dir, 'log.txt')  # 存储运行日志文件
stdout = Logger(logfile)
print('Logging to {:s}'.format(logfile))
sys.stdout = stdout

# 模型设置
feature_extractor = models.resnet34(pretrained=True)
atloc = AtLoc(feature_extractor, droprate=opt.train_dropout, pretrained=True)
if opt.model == 'AtLoc':
    model = atloc
    train_criterion = AtLocCriterion(saq=opt.beta, learn_beta=True)  # 模型训练集误差计算
    val_criterion = AtLocCriterion()  # 模型验证集误差计算
    param_list = [{'params': model.parameters()}]  # 获取模型训练的参数集合
else:
    raise NotImplementedError

# 模型优化
param_list = [{'params': model.parameters()}]  # 模型训练参数集
if hasattr(train_criterion, 'sax') and hasattr(train_criterion, 'saq'):
    print('learn_beta')
    param_list.append({'params': [train_criterion.sax, train_criterion.saq]})

optimizer = torch.optim.Adam(param_list, lr=opt.lr, weight_decay=opt.weight_decay)  # 模型优化器构架

stats_file = osp.join(opt.data_dir, opt.dataset, opt.scene, 'stats.txt')  # 训练数据集统计文件路径设置
stats = np.loadtxt(stats_file)  # 读取训练数据集统计数据
tforms = [transforms.Resize(opt.cropsize)]  # 训练数据集转换处理设置
tforms.append(transforms.RandomCrop(opt.cropsize))  # 设置为随机裁剪图片数据集

# 图片数据增强处理
if opt.color_jitter > 0:
    assert opt.color_jitter <= 1.0
    print('Using Colorjitter data augementation')
    # 对原始图像的亮度，对比度，饱和度以及色调进行增强
    tforms.append(transforms.ColorJitter(brightness=opt.color_jitter, contrast=opt.color_jitter,
                                         saturation=opt.color_jitter, hue=0.5))
else:
    print('Not Using ColorJitter')

# ToTensor就是将一个PIL image转换成一个Tensor,PIL是(H，W，C)的形式，范围是[0,255],而Tensor是(C，H，W)的形式,范围是[0,1]
tforms.append(transforms.ToTensor())
tforms.append(transforms.Normalize(mean=stats[0], std=np.sqrt(stats[1])))  # 根据训练集的统计数据进行数据归一化处理
data_transform = transforms.Compose(tforms)  # 把数据变换操作的多个步骤整合在一起操作

# 自定义自己的transform策略,方法就是使用transforms.Lambda()
target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())  # 将x转换为float类型的tensor

# 加载测试数据集
kwargs = dict(scene=opt.scene, data_path=opt.data_dir, transform=data_transform, target_transform=target_transform,
              seed=opt.seed)

if opt.model == 'AtLoc':
    if opt.dataset == '7Scenes':
        train_set = SevenScenes(train=True, **kwargs)  # 获取测试数据集
        val_set = SevenScenes(train=False, **kwargs)  # 获取验证数据集
    elif opt.dataset == 'RobotCar':
        train_set = RobotCar(train=True, **kwargs)
        val_set = RobotCar(train=False, **kwargs)
    else:
        raise NotImplementedError
else:
    raise NotImplementedError

# num_workers：使用的子进程数，0为不使用多进程
# 是否将tensor数据复制到CUDA pinned memory中，pin memory中的数据转到GPU中会快一些
kwargs = {'num_workers': opt.nThreads, 'pin_memory': True} if cuda else {}

# Dataset负责表示数据集，它可以每次使用__getitem__返回一个样本
# 而torch.utils.data.Dataloader提供了对batch的处理，如shuffle等
# Dataset被封装在了Dataloader中
train_loader = DataLoader(train_set, batch_size=opt.batchsize, shuffle=True, **kwargs)
val_loader = DataLoader(val_set, batch_size=opt.batchsize, shuffle=False, **kwargs)

model.to(device)  # 将模型加载到指定的gpu设备上
train_criterion.to(device)  # 将测试集损失函数模块加载到指定的gpu设备上
val_criterion.to(device)  # 将验证集损失函数模块加载到指定的gpu设备上

total_steps = opt.steps
# 模型运行结果可视化
writer = SummaryWriter(log_dir=opt.runs_dir)  # 建立一个保存结果数据变量writer
experiment_name = opt.exp_name
for epoch in range(opt.epochs):
    if epoch % opt.val_freq == 0 or epoch == (opt.epochs - 1):
        val_batch_time = AverageMeter()
        val_loss = AverageMeter()
        model.eval()
        end = time.time()
        val_data_time = AverageMeter()

        for batch_idx, (val_data, val_target) in enumerate(val_loader):
            val_data_time.updata(time.time() - end)  # 计算花费时间
            val_data_var = Variable(val_data, requires_grad=False)  # variable是tensor的外包装
            val_target_var = Variable(val_target, requires_grad=False)
            val_data_var = val_data_var.to(device)  # 加载到计算设备中
            val_target_var = val_target_var.to(device)

            with torch.set_grad_enabled(False):
                val_output = model(val_data_var)
                val_loss_tmp = val_criterion(val_output, val_target_var)
                val_loss_tmp = val_loss_tmp.item()

            val_loss.updata(val_loss_tmp)  # 更新损失函数值
            val_batch_time.updata(time.time() - end)

            writer.add_scalar('val_error', val_loss_tmp, total_steps)  # 模型运算结果统计（每total_steps记录一次val_error）
            if batch_idx % opt.print_freq == 0:
                print(
                    'Val {:s}: Epoch {:d}\tBatch {:d}/{:d}\tData time {:.4f} ({:.4f})\tBatch time {:.4f} ({:.4f})\tLoss {:f}'.format(
                        experiment_name, epoch, batch_idx, len(val_loader) - 1, val_data_time.val, val_data_time.avg,
                        val_batch_time.val, val_batch_time.avg, val_loss_tmp))
            end = time.time()

        print('Val {:s}:Epoch {:d}, val_loss {:f}'.format(experiment_name, epoch, val_loss.avg))

        if epoch % opt.save_feq == 0:
            filename = osp.join(opt.models_dir, 'epoch_{:03d}.path.tar'.format(epoch))  # 模型权重参数存储路径
            checkpoint_dict = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                               'optim_state_dict': optimizer.state_dict(),
                               'criterion_state_dict': train_criterion.state_dict()}
            torch.save(checkpoint_dict, filename)  # 保存模型参数
            print('Epoch {:d} checkpoint saved for {:s}'.format(epoch, experiment_name))

    model.train()  # 实例化的model指定train
    train_data_time = AverageMeter()
    train_batch_time = AverageMeter()
    end = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        train_data_time.updata(time.time() - end)

        data_var = Variable(data, requires_grad=True)
        target_var = Variable(target, requires_grad=False)
        data_var = data_var.to(device)
        target_var = target_var.to(device)

        with torch.set_grad_enabled(True):  # 计算梯度
            output = model(data_var)
            loss_tmp = train_criterion(output, target_var)

        loss_tmp.backward()  # 反向传播
        optimizer.step()  # 参数优化更新
        optimizer.zero_grad()  # 即将梯度初始化为零（因为一个batch的loss关于weight的导数是所有sample的loss关于weight的导数的累加和）

        train_batch_time.updata(time.time() - end)
        writer.add_scalar('train_err', loss_tmp.item(), total_steps)
        if batch_idx % opt.print_freq == 0:
            print(
                'Train {:s}: Epoch {:d}\tBatch {:d}/{:d}\tData time {:.4f} ({:.4f})\tBatch time {:.4f} ({:.4f})\tLoss {:f}' \
                    .format(experiment_name, epoch, batch_idx, len(train_loader) - 1, train_data_time.val,
                            train_data_time.avg, train_batch_time.val, train_batch_time.avg, loss_tmp.item()))

        end = time.time()

writer.close()
