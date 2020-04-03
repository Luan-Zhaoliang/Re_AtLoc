# -- coding: utf-8 --
# @Time : 01/04/2020 20:21
# @Author : LZL
# @Email: z.luan@qmul.ac.uk
# @File : eval.py

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 保证程序中的GPU序号是和硬件中的序号是相同的
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"  # 设置使用的gpu设备资源

import torch
import os.path as osp
import numpy as np
import matplotlib
import sys

DISPLAY = 'DISPLAY' in os.environ
if not DISPLAY:
    matplotlib.use('Agg')  # 设置后端，Matplotlib绘图并保存图像但不显示图形
import matplotlib.pyplot as plt

from tools.options import Options
from network.atloc import AtLoc
from torchvision import transforms, models
from tools.utils import quaternion_angular_error, qexp, load_state_dict
from torch.utils.data import DataLoader
from data.dataloaders import SevenScenes, RobotCar
from torch.autograd import Variable

# 配置运行环境
opt = Options().parse()  # 参数命令解析
cuda = torch.cuda.is_available()  # 判断是否有可用gpu设备
device = "cuda" + ",".join(str(i) for i in opt.gpus) if cuda else "cpu"  # 获取gpu或cpu设备信息

# 模型设置
feature_extractor = models.resnet34(pretrained=False)  # resnet34模型作为特征提取器
atloc = AtLoc(feature_extractor, droprate=opt.test_dropout, pretrained=False)  # atloc模型实例构建
if opt.model == 'AtLoc':
    model = atloc
else:
    raise NotImplementedError

# model.eval(),pytorch会自动把BN和DropOut固定住，不会取平均，而是用训练好的值
# 不然的话，一旦test的batch_size过小，很容易就会被BN层导致生成图片颜色失真极大
model.eval()  # 实例化的model指定为test模式

# 模型损损失值计算
t_criterion = lambda t_pred, t_gt: np.linalg.norm(t_pred - t_gt)
q_criterion = quaternion_angular_error

stats_file = osp.join(opt.data_dir, opt.dataset, opt.scene, 'stats.txt')
stats = np.loadtxt(stats_file)  # 获取统计数据"stats.txt"，转换为ndarray数据类型

# 图像转换操作
# transforms.Compose将归一化(normalize)，尺寸剪裁(resize)，ToTensor等步骤整合在一起执行
# Resize把给定的图片resize到given size
# Normalize就是将一个tensor image根据其均值和方差进行归一化
# ToTensor就是将一个PIL image转换成一个Tensor,PIL是(H，W，C)的形式，范围是[0,255],而Tensor是(C，H，W)的形式,范围是[0,1]
data_transform = transforms.Compose(
    [transforms.Resize(opt.cropsize), transforms.CenterCrop(opt.cropsize), transforms.ToTensor(),
     transforms.Normalize(mean=stats[0], std=np.sqrt((stats[1])))])

# 自定义自己的transform策略,方法就是使用transforms.Lambda()
target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())  # 将x转换为float类型的tensor

# read mean and stdev for un-normalizing predictions
pose_stats_file = osp.join(opt.data_dir, opt.dataset, opt.scene, 'pose_stats.txt')
pose_m, pose_s = np.loadtxt(pose_stats_file)  # 获取pose统计文件中的均值和标准差

# 加载测试数据集
kwargs = dict(scene=opt.scene, data_path=opt.data_dir, train=False, transform=data_transform,
              target_transform=target_transform, seed=opt.seed)

if opt.model == 'AtLoc':
    if opt.dataset == '7Scenes':
        data_set = SevenScenes(**kwargs)  # 获取处理后的7Scenes数据
    elif opt.dataset == 'RobotCar':
        data_set = RobotCar(**kwargs)  # 获取处理后的RobotCar数据
else:
    raise NotImplementedError

L = len(data_set)

# num_workers：使用的子进程数，0为不使用多进程
# 是否将tensor数据复制到CUDA pinned memory中，pin memory中的数据转到GPU中会快一些
kwargs = {'num_workers': opt.nThreads, 'pin_memory': True} if cuda else {}

# Dataset负责表示数据集，它可以每次使用__getitem__返回一个样本
# 而torch.utils.data.Dataloader提供了对batch的处理，如shuffle等
# Dataset被封装在了Dataloader中
loader = DataLoader(data_set, batch_size=1, shuffle=False, **kwargs)

pred_poses = np.zeros((L, 7))  # save all predicted poses
targ_poses = np.zeros((L, 7))  # save all target poses

# 加载训练好的权重参数文件
model.to(device)  # 将模型加载到指定的gpu设备上
weights_filename = osp.expanduser(opt.weights)  # 把path中包含的"~"和"~user"转换成用户目录
if osp.isfile(weights_filename):
    checkpoint = torch.load(weights_filename, map_location=device)  # 将权重文件加载到指定的设备上
    load_state_dict(model, checkpoint['model_state_dict'])
    print('Lodaded weights from {:s}'.format(weights_filename))
else:
    print('Could not load weights from {:s}'.format(weights_filename))
    sys.exit(-1)  # 退出程序

# inference loop
for idx, (data, target) in enumerate(loader):
    if idx % 200 == 0:
        print('Image {:d}/{:d}'.format(idx, len(loader)))

    # output : 1 x 6
    data_var = Variable(data, requires_grad=False)  # variable是tensor的外包装
    data_var = data_var.to(device)  # data_var加载到指定设备上

    with torch.set_grad_enabled(False):
        output = model(data_var)
    s = output.size()
    output = output.cpu().data.numpy().reshape((-1, s[-1]))  # 模型计算后输出值
    target = target.numpy().reshape((-1, s[-1]))  # 目标值

    # normalize the predicted quaternions
    q = [qexp(p[3:]) for p in output]
    output = np.hstack((output[:, :3], np.asarray(q)))
    q = [qexp(p[3:]) for p in target]
    target = np.hstack((target[:, :3], np.asarray(q)))

    # un-normalized the predicted and target translations
    output[:, :3] = (output[:, :3] * pose_s) + pose_m
    target[:, :3] = (target[:, :3] * pose_s) + pose_m

    # take the middle prediction
    pred_poses[idx, :] = output[len(output) // 2]
    targ_poses[idx, :] = target[len(target) // 2]

# 计算损失函数
# zip()函数将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表，也就是返回对应的predict和target来计算误差
t_loss = np.asarray([t_criterion(p, t) for p, t in zip(pred_poses[:, :3], targ_poses[:, :3])])
q_loss = np.asarray([q_criterion(p, t) for p, t in zip(pred_poses[:, 3:], targ_poses[:, 3:])])
print(
    'Error in translation: media {:3.2f} m, mean {:3.2f} m \nError in rotation: median {:3.2f} degrees, mean {:3.2f} degree'.format(
        np.median(t_loss), np.mean(t_loss), np.median(q_loss), np.mean(q_loss)))

fig = plt.figure()
real_pose = (pred_poses[:, :3] - pose_m) / pose_s
gt_pose = (targ_poses[:, :3] - pose_m) / pose_s

plt.plot(gt_pose[:, 1], real_pose[:, 0], color='black')
plt.plot(real_pose[:, 1], real_pose[:, 0], color='red')

plt.xlabel('x [m]')
plt.ylabel('y [m]')

plt.plot(gt_pose[0, 1], gt_pose[0, 0], 'y*', markersize=15)
plt.show(block=True)
image_filename = osp.join(osp.expanduser(opt.results_dir), '{:s}.png'.format(opt.exp_name))
fig.savefig(image_filename)
