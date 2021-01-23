import numpy as np
import os
import cv2
import re
import csv
import glob
import random
import pickle
from geometry.transformation import *
import tensorflow as tf
from scipy import interpolate
import argparse
import random
import numpy
import sys


fx = 517.3
fy = 516.5
cx = 318.6
cy = 255.3
intrinsics = np.array([fx, fy, cx, cy])

# 时间轴对齐
def associate_frames(image_times, depth_times, pose_times):
    associations = []
    for i, t in enumerate(image_times):
        j = np.argmin(np.abs(depth_times - t))
        k = np.argmin(np.abs(pose_times - t))
        associations.append((i, j, k))
    return associations

_EPS = numpy.finfo(float).eps * 4.0

def transform44(l):
    """
    Generate a 4x4 homogeneous transformation matrix from a 3D point and unit quaternion.
    
    Input:
    l -- tuple consisting of (stamp,tx,ty,tz,qx,qy,qz,qw) where
         (tx,ty,tz) is the 3D position and (qx,qy,qz,qw) is the unit quaternion.
         
    Output:
    matrix -- 4x4 homogeneous transformation matrix
    """
    t = l[1:4]
    q = numpy.array(l[4:8], dtype=numpy.float64, copy=True)
    nq = numpy.dot(q, q)
    if nq < _EPS:
        return numpy.array((
        (                1.0,                 0.0,                 0.0, t[0])
        (                0.0,                 1.0,                 0.0, t[1])
        (                0.0,                 0.0,                 1.0, t[2])
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=numpy.float64)
    q *= numpy.sqrt(2.0 / nq)
    q = numpy.outer(q, q)
    return numpy.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], t[0]),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], t[1]),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], t[2]),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=numpy.float64)


factor = 5000 # for the 16-bit PNG files
def fill_depth(depth):
    x, y = np.meshgrid(np.arange(depth.shape[1]).astype("float32"),
                       np.arange(depth.shape[0]).astype("float32"))
    xx = x[depth > 0]
    yy = y[depth > 0]
    zz = depth[depth > 0]

    grid = interpolate.griddata((xx, yy), zz.ravel(),
                                (x, y), method='nearest')
    return grid

class TUM_RGBD:
    """主要用来进行数据的加载与查找
    """
    def __init__(self, dataset_path, n_frames=4, r=6):
        self.dataset_path = dataset_path
        self.n_frames = n_frames
        self.height = 480
        self.width = 640
        self.build_dataset_index(r=r)
    # 获取数据长度
    def __len__(self):
        return len(self.dataset_index)
    # 获取数据长度
    def shape(self):
        return [self.n_frames, self.height, self.width]
    # 获取数据
    def __getitem__(self, index):
        data_blob = self.dataset_index[index]
        num_frames = data_blob['n_frames']
        num_samples = self.n_frames - 1

        frameid = data_blob['id']
        keyframe_index = num_frames // 2 # 选取中间帧作为关键帧

        inds = np.arange(num_frames)
        inds = inds[~np.equal(inds, keyframe_index)]
        
        inds = np.random.choice(inds, num_samples, replace=False)
        inds = [keyframe_index] + inds.tolist()
        # 读取图像
        images = []
        for i in inds:
            image = cv2.imread(data_blob['images'][i])
            image = cv2.resize(image, (640, 480))
            images.append(image)

        poses = []
        for i in inds:
            poses.append(data_blob['poses'][i])
        # 转换图像和深度信息
        images = np.stack(images, axis=0).astype(np.uint8)
        poses = np.stack(poses, axis=0).astype(np.float32)
        # 获取深度信息
        depth_file = data_blob['depth']
        # 读取深度信息
        depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
        
        depth = (depth.astype(np.float32)) / 1000.0
        filled = fill_depth(depth)
        
        K = data_blob['intrinsics']
        # 相机内参，转换为向量矩阵
        kvec = np.stack([K[0,0], K[1,1], K[0,2], K[1,2]], axis=0)

        depth = depth[...,None]
        return images, poses, depth, filled, filled, kvec, frameid


    def __iter__(self):
        random.shuffle(self.dataset_index)
        while 1:
            self.perm = np.arange(len(self.dataset_index))
            np.random.shuffle(self.perm)
            for i in self.perm:
                yield self.__getitem__(i)
    # 加载数据文件夹
    def _load_scan(self, sequence_name):
        """加载对应的image和基础文件信息

        Args:
            scan str: 扫描的目标文件夹

        Returns:
            str: 最终为文件属性
        """
        sequence_dir = os.path.join(self.dataset_path, sequence_name)
        # 创建序列化文件
        datum_file = os.path.join(scan_path, 'pickle-TUM.pkl')
        # 如果序列化文件不存在就进行创建
        if not os.path.isfile(datum_file):
            # 获取图像列表
            image_list = os.path.join(sequence_dir, 'rgb.txt')
            # 获取深度列表
            depth_list = os.path.join(sequence_dir, 'depth.txt')
            # 获取位姿列表
            pose_list = os.path.join(sequence_dir, 'groundtruth.txt')

            # 图像数据
            images = np.loadtxt(image_list, delimiter=' ', dtype=np.unicode_, skiprows=3)
            # 深度数据
            depths = np.loadtxt(depth_list, delimiter=' ', dtype=np.unicode_, skiprows=3)
            # 位姿数据
            try:
                poses = np.loadtxt(pose_list, delimiter=' ', dtype=np.float64, skiprows=3)
            except:
                poses = np.zeros((len(images), 7))

            depth_intrinsics = intrinsics.copy()
            color_intrinsics = intrinsics.copy()
            # 写入序列化数据
            datum = images, depths, poses, color_intrinsics, depth_intrinsics
            pickle.dump(datum, open(datum_file, 'wb'))
            
        else:
            datum = pickle.load(open(datum_file, 'rb'))

        return datum


    def build_dataset_index(self, r=4, skip=12):
        self.dataset_index = []
        data_id = 0
        # 访问数据文件夹，并列举所有数据文件夹
        for scan in sorted(os.listdir(self.dataset_path)):

            # 访问数据文件夹，构造对应的数据
            images, depths, poses, color_intrinsics, depth_intrinsics = self._load_scan(scan)

            for i in range(r, len(images)-r, skip):
                # some poses in scannet are nans
                if np.any(np.isnan(poses[i-r:i+r+1])):
                    continue

                training_example = {}
                training_example['depth'] = depths[i]
                training_example['images'] = images[i-r:i+r+1]
                training_example['poses'] = poses[i-r:i+r+1]
                training_example['intrinsics'] = depth_intrinsics
                training_example['n_frames'] = 2*r+1
                training_example['id'] = data_id

                self.dataset_index.append(training_example)
                data_id += 1

    def test_set_iterator(self):

        test_frames = np.loadtxt('data/scannet/scannet_test.txt', dtype=np.unicode_)
        test_data = []

        for i in range(0, len(test_frames), 4):
            test_frame_1 = str(test_frames[i]).split('/')
            test_frame_2 = str(test_frames[i+1]).split('/')
            scan = test_frame_1[3]

            imageid_1 = int(re.findall(r'frame-(.+?).color.jpg', test_frame_1[-1])[0])
            imageid_2 = int(re.findall(r'frame-(.+?).color.jpg', test_frame_2[-1])[0])            
            test_data.append((scan, imageid_1, imageid_2))

        # random.shuffle(test_data)        
        for (scanid, imageid_1, imageid_2) in test_data:

            scandir = os.path.join(self.dataset_path, scanid)
            num_frames = len(os.listdir(os.path.join(scandir, 'color')))

            images = []

            # we need to include imageid_2 and imageid_1 to compare to BA-Net poses,
            # then sample remaining 6 frames uniformly
            dt = imageid_2 - imageid_1
            s = 3

            for i in [0, dt, -3*s, -2*s, -s, s, 2*s, 3*s]:
                otherid = min(max(1, i+imageid_1), num_frames-1)
                image_file = os.path.join(scandir, 'color', '%d.jpg'%otherid)
                image = cv2.imread(image_file)
                image = cv2.resize(image, (640, 480))
                images.append(image)

            depth_file = os.path.join(scandir, 'depth', '%d.png'%imageid_1)
            depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
            depth = (depth/1000.0).astype(np.float32)

            pose1 = np.loadtxt(os.path.join(scandir, 'pose', '%d.txt'%imageid_1), delimiter=' ')
            pose2 = np.loadtxt(os.path.join(scandir, 'pose', '%d.txt'%imageid_2), delimiter=' ')
            pose1 = np.linalg.inv(pose1)
            pose2 = np.linalg.inv(pose2)
            pose_gt = np.dot(pose2, np.linalg.inv(pose1))

            depth_intrinsics = os.path.join(scandir, 'intrinsic/intrinsic_depth.txt')
            K = np.loadtxt(depth_intrinsics, delimiter=' ')
            fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]

            images = np.stack(images, axis=0).astype(np.uint8)
            depth = depth.astype(np.float32)
            intrinsics = np.array([fx, fy, cx, cy], dtype=np.float32)

            data_blob = {
                'images': images,
                'depth': depth,
                'pose': pose_gt,
                'intrinsics': intrinsics,
            }

            yield data_blob


    # def iterate_sequence(self, scan):
    #     # 扫描路径
    #     scan_path = os.path.join(self.dataset_path, scan)
    #     imfiles = glob.glob(os.path.join(scan_path, 'pose', '*.txt'))
    #     ixs = sorted([int(os.path.basename(x).split('.')[0]) for x in imfiles])

    #     K = np.loadtxt(os.path.join(scan_path, 'intrinsic', 'intrinsic_depth.txt'), delimiter=' ')
    #     fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    #     intrinsics = np.array([fx, fy, cx, cy], dtype=np.float32)

    #     images = []
    #     for i in ixs:
    #         imfile = os.path.join(scan_path, 'color', '%d.jpg'%i)
    #         image = cv2.imread(imfile)
    #         image = cv2.resize(image, (640, 480))
    #         yield image, intrinsics
    # 读取迭代器列表
    def iterate_sequence(self, sequence_name, matrix=False):
        """returns list of images, depths, and poses 返回地址位姿"""
        sequence_dir = os.path.join(self.dataset_path, sequence_name)
        image_list = os.path.join(sequence_dir, 'rgb.txt')
        depth_list = os.path.join(sequence_dir, 'depth.txt')
        pose_list = os.path.join(sequence_dir, 'groundtruth.txt')
        # 图像数据
        image_data = np.loadtxt(image_list, delimiter=' ', dtype=np.unicode_, skiprows=3)
        # 深度数据
        depth_data = np.loadtxt(depth_list, delimiter=' ', dtype=np.unicode_, skiprows=3)

        try:
            pose_data = np.loadtxt(pose_list, delimiter=' ', dtype=np.float64, skiprows=3)
        except:
            pose_data = np.zeros((len(image_data), 7))
            secret = True
        # 获取相机参数矩阵
        intrinsics_mat = intrinsics.copy()

        images = []
        for (tstamp, image_file) in image_data:
            image_file = os.path.join(sequence_dir, image_file)
            image = cv2.imread(image_file)

            yield image, intrinsics_mat

        #     images.append(image)

        # depths = []
        # for (_, depth_file) in depth_data:
        #     depth_file = os.path.join(sequence_dir, depth_file)
        #     depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
        #     depth = depth.astype(np.float32) / 5000.0
        #     depths.append(depth)

        # traj_gt = []
        # for pose_vec in pose_data:
        #     if matrix:
        #         traj_gt.append(transform44(pose_vec))
        #     else:
        #         traj_gt.append(pose_vec)

        # image_times = image_data[:,0].astype(np.float64)
        # depth_times = depth_data[:,0].astype(np.float64)
        # pose_times = pose_data[:,0].astype(np.float64)
        # indicies = associate_frames(image_times, depth_times, pose_times)

        # rgbd_images = []
        # rgbd_depths = []
        # timestamps = []
        # for (img_ix, depth_ix, pose_ix) in indicies:
        #     timestamps.append(image_times[img_ix])
        #     rgbd_images.append(images[img_ix])
        #     rgbd_depths.append(depths[depth_ix])
            
        # timestamps = np.stack(timestamps, axis=0)
        # rgbd_images = np.stack(rgbd_images, axis=0)
        # rgbd_depths = np.stack(rgbd_depths, axis=0)
        # intrinsics_mat = intrinsics.copy()

        # for img in rgbd_images:
        #     yield img, intrinsics_mat