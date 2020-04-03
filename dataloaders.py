# -- coding: utf-8 --
# @Time : 01/04/2020 20:22
# @Author : LZL
# @Email: z.luan@qmul.ac.uk
# @File : dataloaders.py

import os
import torch
import numpy as np
import pickle
import os.path as osp
import tools.utils

from data.robotcar_sdk.interpolate_poses import interpolate_vo_poses, interpolate_ins_poses
from data.robotcar_sdk.image import load_image
from data.robotcar_sdk.camera_model import CameraModel
from tools.utils import prosess_poses, calc_vos_simple,load_images
from torch.utils import data
from functools import partial


# 继承Dataset类
class SevenScenes(data.Dataset):
    # 初始化，定义数据内容
    def __init__(self, scene, data_path, train, transform=None, target_transform=None, mode=0, seed=0, real=False,
                 skip_images=False, vo_lib='orbslam'):
        self.mode = mode
        self.transform = transform  # 图像变换处理
        self.target_transform = target_transform
        self.skip_images = skip_images  # 间隔skip_images取图片数据
        np.random.seed(seed)  # 生成相同的随机数

        # 文件路径设置
        data_dir = osp.join(data_path, '7Scenes', scene)

        # 判断是train还是test来选择加载的文件
        if train:
            split_file = osp.join(data_dir, 'train_split.txt')
        else:
            split_file = osp.join(data_dir, 'test_split.txt')

        with open(split_file, 'r') as f:
            seqs = [int(l.split('sequence')[-1]) for l in f if not l.startswith('#')]  # 获取数据文件名称序列

        # 读取poses数据和设置图像文件名称
        self.c_imgs = []
        self.d_imgs = []
        self.get_idx = np.empty((0,), dtype=np.int)
        ps = {}
        vo_stats = {}
        gt_offset = int(0)
        for seq in seqs:
            seq_dir = osp.join(data_dir, 'seq-{:02d}'.format(seq))  # 路径设置
            p_filenames = [n for n in os.listdir(osp.join(seq_dir, '.')) if n.find('pose') >= 0]
            if real:
                pose_file = osp.join(data_dir, '{:s}_poses'.format(vo_lib), 'seq-{:02d}.txt'.format(seq))
                pss = np.loadtxt(pose_file)  # 读取出pose文件中的数据
                frame_idx = pss[:, 0].astype(np.int)  # 转换为int类型
                if vo_lib == 'libviso2':
                    frame_idx -= 1
                ps[seq] = pss[:, 1:13]
                vo_stats_filename = osp.join(seq_dir, '{:s}_vo_stats.pkl'.format(vo_lib))
                with open(vo_stats_filename, 'rb') as f:
                    vo_stats[seq] = pickle.load(f)  # 将f中的对象序列化读出
            else:
                frame_idx = np.array(range(len(p_filenames)), dtype=np.int)
                pss = [np.loadtxt(osp.join(seq_dir, 'frame-{:06d}.pose.txt'.format(i))).flatten()[:12] for i in
                       frame_idx]
                ps[seq] = np.asarray(pss)  # 将pose数据存储为字典类型
                vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}

            self.get_idx = np.hstack((self.get_idx, gt_offset + frame_idx))
            gt_offset += len(p_filenames)
            c_imgs = [osp.join(seq_dir, 'frame-{:06d}.color.png'.format(i)) for i in frame_idx]
            d_imgs = [osp.join(seq_dir, 'frame-{:06d}.depth.png'.format(i)) for i in frame_idx]
            self.c_imgs.extend(c_imgs)  # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）
            self.d_imgs.extend(d_imgs)

        pose_stats_filename = osp.join(data_dir, 'pose_stats.txt')
        if train and not real:
            mean_t = np.zeros(3)  # optionally, use the ps dictionary to calculate stats
            std_t = np.zeros(3)
            np.savetxt(pose_stats_filename, np.vstack(mean_t, std_t), fmt='%8.7f')  # 将array按照一定格式保存为txt文件
        else:
            mean_t, std_t = np.loadtxt(pose_stats_filename)

        # 转变pose的形式为：translation + log quaternion
        self.poses = np.empty((0, 6))
        for seq in seqs:
            pss = prosess_poses(poses_in=ps[seq], mean_t=mean_t, std_t=std_t, align_R=vo_stats[seq]['R'],
                                align_t=vo_stats[seq]['t'], align_s=vo_stats[seq]['s'])  # 获取处理后的pose数据
            self.poses = np.vstack((self.poses, pss))  # np.vstack():在竖直方向上堆叠

    #  获取数据内容
    def __getitem__(self, index):
        if self.skip_images:
            img = None
            pose = self.poses[index]
        else:
            if self.mode == 0:
                img = None
                while img is None:
                    img = load_images(self.c_imgs[index])  # 获取color image数据
                    pose = self.poses[index]  # 获取pose数据
                    index += 1
                index -= 1
            elif self.mode == 1:
                img = None
                while img is None:
                    img = load_images(self.d_imgs[index])  # 获取depth image数据
                    pose = self.poses[index]  # 获取pose数据
                    index += 1
                index -= 1
            elif self.mode == 2:
                c_img = None
                d_img = None
                while (c_img is None) or (d_img is None):
                    c_img = load_images(self.c_imgs[index])  # 获取color image数据
                    d_img = load_images(self.d_imgs[index])  # 获取depth image数据
                    pose = self.poses[index]
                    index += 1
                img = [c_img, d_img]
                index -= 1
            else:
                raise Exception('Wrong mode {:d}'.format(self.mode))

        if self.target_transform is not None:
            pose = self.target_transform(pose)

        if self.skip_images:
            return img, pose

        if self.transform is not None:
            if self.mode == 2:
                img = [self.transform(i) for i in img]
            else:
                img = self.transform(img)

        return img, pose

    # 返回数据集大小
    def __len__(self):
        return self.poses.shape[0]


# 继承Dataset类
class RobotCar(data.Dataset):
    # 初始化，定义数据内容
    def __init__(self, scene, data_path, train, transform=None, target_transform=None, real=False, skip_images=False,
                 seed=7, undisort=False, vo_lib='stereo'):
        np.random.seed(seed)
        self.transform = transform  # 图像变换处理
        self.target_transform = target_transform
        self.skip_images = skip_images
        self.undistort = undisort  # 图像畸变设置

        # 文件路径设置
        data_dir = osp.join(data_path, 'RobotCar', scene)

        # 判断是train还是test来选择加载的文件
        if train:
            split_filename = osp.join(data_dir, 'train_split.txt')
        else:
            split_filename = osp.join(data_dir, 'test_split.txt')
        with open(split_filename, 'r') as f:
            seqs = [l.rstrip() for l in f if not l.startswith('#')]  # 获取数据文件名称序列

        ps = {}  # 存储图片pose数据
        ts = {}  # 存储图片timestamps数据
        vo_stats = {}
        self.imgs = []
        for seq in seqs:
            seq_dir = osp.join(data_dir, seq)
            # 获取图像timestamps数据
            ts_filename = osp.join(seq_dir, 'stereo.timestamps')
            with open(ts_filename, 'r') as f:
                ts[seq] = [int(l.rstrip().split(' ')[0]) for l in f]  # 获取timestamps序列数据

            if real:  # poses from integration of VOs
                if vo_lib == 'stereo':
                    vo_filename = osp.join(seq_dir, 'vo', 'vo.csv')  # vo.csv为相对位姿数据
                    # Interpolate poses from visual odometry
                    p = np.asarray(interpolate_vo_poses(vo_filename, ts[seq], ts[seq][0]))
                elif vo_lib == 'gps':
                    vo_filename = osp.join(seq_dir, 'gps', 'gps_ins.csv')
                    # Interpolate poses from INS
                    p = np.asarray(interpolate_ins_poses(vo_filename, ts[seq], ts[seq][0]))
                else:
                    raise NotImplementedError
                vo_stats_filename = osp.join(seq_dir, '{:s}_vo_stats.pkl'.format(vo_lib))
                with open(vo_stats_filename, 'r') as f:
                    vo_stats[seq] = pickle.load(f)  # 将f中的对象序列化读出
                ps[seq] = np.reshape(p[:, :3, :], (len(p), -1))
            else:
                pose_filename = osp.join(seq_dir, 'gps', 'ins.csv')  # 获取ins.cv文件
                # Interpolate poses from visual odometry
                p = np.asarray(interpolate_ins_poses(pose_filename, ts[seq], ts[seq][0]))
                ps[seq] = np.reshape(p[:, :3, :], (len(p), -1))
                vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}

            # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）
            self.imgs.extend([osp.join(seq_dir, 'stereo', 'centre_processed', '{:d}.png'.format(t)) for t in ts[seq]])

        # read/save pose normalization information
        poses = np.empty((0, 12))
        for p in ps.values():
            poses = np.vstack((poses, p))
        pose_stats_filename = osp.join(data_dir, 'pose_stats.txt')  # pose统计数据
        if train and not real:
            mean_t = np.mean(poses[:, [3, 7, 11]], axis=0)  # 计算矩阵均值
            std_t = np.std(poses[:, [3, 7, 11]], axis=0)  # 计算矩阵标准差
            np.savetxt(pose_stats_filename, np.vstack((mean_t, std_t)), fmt='%8.7f')  # 将narray存储到text文件中
        else:
            mean_t, std_t = np.loadtxt(pose_stats_filename)

        # 将pose数据转换为：translation + log quaternion, align, normalize
        self.poses = np.empty((0, 6))
        for seq in seqs:
            # 获取处理后的pose数据
            pss = prosess_poses(poses_in=ps[seq], mean_t=mean_t, std_t=std_t, align_R=vo_stats[seq]['R'],
                                align_t=vo_stats[seq]['t'], align_s=vo_stats[seq]['s'])
            self.poses = np.vstack((self.poses, pss))
        self.gt_idx = np.asarray(range(len(self.poses)))

        # camera model and image loader
        camera_model = CameraModel('/Users/williamed/Desktop/AtLoc-master/data/robotcar_camera_models',
                                   osp.join('stereo', 'centre_processed'))

        # partial是对原始函数的二次封装，是将现有函数的部分参数预先绑定为指定值，从而得到一个新的函数
        self.im_loader = partial(load_image, model=camera_model)

    # 获取数据内容
    def __getitem__(self, index):
        if self.skip_images:
            img = None
            pose = self.poses[index]
        else:
            img = None
            while img is None:
                if self.undistort:
                    # np.uint8将img数据转换为0-255数据类型
                    img = np.uint8(tools.utils.load_images(self.imgs[index], loader=self.im_loader))
                else:
                    img = tools.utils.load_images(self.imgs[index])
                pose = np.float32(self.poses[index])
                index += 1
            index -= 1

        if self.target_transform is not None:
            pose = self.target_transform(pose)

        if self.skip_images:
            return img, pose

        if self.transform is not None:
            img = self.transform(img)

        return img, pose

    # 返回数据集大小
    def __len__(self):
        return len(self.poses)
