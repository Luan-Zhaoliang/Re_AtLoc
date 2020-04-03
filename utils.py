# -- coding: utf-8 --
# @Time : 01/04/2020 07:43
# @Author : LZL
# @Email: z.luan@qmul.ac.uk
# @File : utils.py

import os
import torch
from torch import nn
import transforms3d.quaternions as txq
import numpy as np
import sys

from torchvision.datasets.folder import default_loader
from _collections import OrderedDict


def madirs(paths):
    '''根据paths的类型来创建文件夹'''
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
        else:
            mkdir(path)


def mkdir(path):
    '''检查文件夹是否存在，否则新建'''
    if not os.path.exists(path):
        os.makedirs(path)


def qlog(q):
    if all(q[1:] == 0):
        q = np.zeros(3)
    else:
        q = np.arccos(q[0]) * q[1:] / np.linalg.norm(q[1:])
    return q


def calc_vos_simple(poses):
    """
    calculate the VOs, from a list of consecutive poses
    :param poses: N x T x 7
    :return: N x (T-1) x 7
    """
    vos = []
    for p in poses:
        # unsqueeze(0)表示在第0维上增加1个维度，对数据维度进行扩充
        pvos = [p[i + 1].unsqueeze(0) - p[i].unsqueeze(0) for i in range(len(p) - 1)]
        vos.append(torch.cat(pvos, dim=0))  # 沿着dim连接pvos中的tensor, 所有的tensor必须有相同的size或为empty
    vos = torch.stack(vos, dim=0)  # stack增加新的维度进行堆叠


def prosess_poses(poses_in, mean_t, std_t, align_R, align_t, align_s):
    """
     processes the 1x12 raw pose from dataset by aligning and then normalizing
     :param poses_in: N x 12
     :param mean_t: 3
     :param std_t: 3
     :param align_R: 3 x 3
     :param align_t: 3
     :param align_s: 1
     :return: processed poses (translation + quaternion) N x 7
     """
    poses_out = np.zeros((len(poses_in), 6))
    poses_out[:, 0:3] = poses_in[:, [3, 7, 11]]

    # align
    for i in range(len(poses_out)):
        R = poses_in[i].reshape((3, 4))[:3, :3]
        q = txq.mat2quat(np.dot(align_R, R))
        q *= np.sign(q[0])  # 限制在半球上
        q = qlog(q)
        poses_out[i, 3:] = q
        t = poses_out[i, :3] - align_t
        poses_out[i, :3] = align_s * np.dot(align_R,
                                            t[:, np.newaxis]).squeeze()  # squeeze()从数组的形状中删除单维度条目，即把shape中为1的维度去掉

    # normalize translation
    poses_out[:, :3] -= mean_t
    poses_out[:, :3] /= std_t
    return poses_out


# 给定filename，通过default_loader()函数载入数据
def load_images(filename, loader=default_loader):
    try:
        img = loader(filename)  # 获取数据集
    except IOError as e:
        print('Could not loas image {:s}, IOError: {:s}'.format(filename, e))
        return None
    except:
        print('Could not load image {:s}, unexpected error'.format(filename))
        return None
    return img


def quaternion_angular_error(q1, q2):
    """
    angular error between two quaternions
    :param q1: (4, )
    :param q2: (4, )
    :return:
    """
    d = abs(np.dot(q1, q2))
    d = min(1.0, max(-1.0, d))
    theta = 2 * np.arccos(d) * 180 / np.pi
    return theta


def qexp(q):
    """
    Applies the exponential map to q
    :param q: (3,)
    :return: (4,)
    """
    n = np.linalg.norm(q)  # 数据初始化
    q = np.hstack((np.cos(n), np.sinc(n / np.pi) * q))  # The sinc function is sin(pi*x) / (pi*x).
    return q


def load_state_dict(model, state_dict):  # state_dict 是一个简单的python的字典对象,将每一层与它的对应参数建立映射关系.(如model的每一层的weights及偏置等等)
    """
     Loads a state dict when the model has some prefix before the parameter names
     :param model:
     :param state_dict:
     :return: loaded model
     """
    model_names = [n for n, _ in model.named_parameters()]
    state_names = [n for n in state_dict.keys()]

    # find prefix for the model and state dicts from the first param name
    if model_names[0].find(state_names[0]) >= 0:
        model_prefix = model_names[0].replace(state_names[0], '')  # 将state_names[0]替换为''
        state_prefix = None
    elif state_names[0].find(model_names[0]) >= 0:
        state_prefix = state_names[0].replace(model_names[0], '')
        model_prefix = None
    else:
        model_prefix = model_names[0].split('.')[0]
        state_prefix = state_names[0].split('.')[0]

    new_state_dict = OrderedDict()  # 对字典对象中元素进行排序
    for k, v in state_dict.items():
        if state_prefix is None:
            k = model_prefix + k
        else:
            k = k.replace(state_prefix, model_prefix)
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict)  # 返回的是一个OrderDict，存储了网络结构的名字和对应的参数，下面看看源代码如何实现的


# 控制程序运行输出
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.termimal = sys.stdout
        self.log = open(filename, "w")

    def delink(self):
        self.log.close()

    def writeTerminalOnly(self, message):
        self.termimal.write(message)

    def write(self, message):
        self.termimal.write(message)
        self.log.write(message)

    def flush(self):
        pass


# 模型损失函数模块
class AtLocCriterion(nn.Module):
    # 初始化
    def __init__(self, t_loss_fn=nn.L1Loss(), q_loss_fn=nn.L1Loss(), sax=0.0, saq=0.0, learn_beta=False):
        super(AtLocCriterion, self).__init__()  # nn.Module的子类函数必须在构造函数中执行父类的构造函数,等价与nn.Module.__init__()
        self.t_loss_fn = t_loss_fn
        self.q_loss_fn = q_loss_fn

        # 首先可以把Parameter函数理解为类型转换函数，将一个不可训练的类型Tensor转换成可以训练的类型parameter并将这个parameter
        # 绑定到这个module里面(net.parameter()中就有这个绑定的parameter，所以在参数优化的时候可以进行优化的)所以经过类型转换这个
        # self.sax变成了模型的一部分，成为了模型中根据训练可以改动的参数了
        # 使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。
        self.sax = nn.Parameter(torch.Tensor([sax]), requires_grad=learn_beta)
        self.saq = nn.Parameter(torch.Tensor([saq]), requires_grad=learn_beta)

    def forward(self, pred, targ):
        loss = torch.exp(-self.sax) * self.t_loss_fn(pred[:, :3], targ[:, :3]) + self.sax + \
               torch.exp(-self.saq) * self.q_loss_fn(pred[:, 3:], targ[:, 3:]) + self.saq

        return loss


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def updata(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
