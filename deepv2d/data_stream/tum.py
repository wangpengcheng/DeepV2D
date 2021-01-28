import numpy as np
import os
import cv2
import re
import csv
import glob
import random
import pickle
from geometry.transformation import *
from scipy import interpolate
import argparse
import random
import numpy
import sys
from utils.tum_associate import *

fx = 517.3
fy = 516.5
cx = 318.6
cy = 255.3
factor = 5000.0 # for the 16-bit PNG files
intrinsics = np.array([fx, fy, cx, cy],dtype=np.float32)

def get_TUM_data(data_path):
    """读取tum中的数据

    Args:
        data_path ([str]): 数据存在的路径 

    Returns:
        [type]: [description]
    """
    sum_data_file_name = "rgb_depth_ground.txt"
    data_file_path = os.path.join(data_path,sum_data_file_name)
    print("==== {} ======".format(data_file_path))
    # 不存在综合数据就进行创建
    if not os.path.isfile(data_file_path):
        # 获取图像列表
        image_list = os.path.join(data_path, 'rgb.txt')
        # 获取深度列表
        depth_list = os.path.join(data_path, 'depth.txt')
        # 获取位姿列表
        pose_list = os.path.join(data_path, 'groundtruth.txt')
        # 执行数据合并文件
        tum_associate(image_list,depth_list,pose_list,data_file_path)

    # 读取综合数据文件
    # 文件样例:
    # 1341841310.82 rgb/1341841310.821702.png 1341841310.82 depth/1341841310.822401.png 1341841310.81 -2.7366 0.1278 1.2413 0.7256 -0.5667 0.2209 -0.3217
    #
    images,depths,poses = get_data_from_sum_file(data_file_path)
    images = [data_path+'/'+i for i in images ]
    depths = [data_path+'/'+i for i in depths ]
    return images,depths,poses

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



def fill_depth(depth):
    x, y = np.meshgrid(np.arange(depth.shape[1]).astype("float32"),
                       np.arange(depth.shape[0]).astype("float32"))
    xx = x[depth > 0]
    yy = y[depth > 0]
    zz = depth[depth > 0]

    grid = interpolate.griddata((xx, yy), zz.ravel(),
                                (x, y), method='nearest')
    return grid
# 将四元数组转换为旋转矩阵
def quat2rotm(q):
    """Convert quaternion into rotation matrix """
    q /= np.sqrt(np.sum(q**2)) 
    x, y, z, s = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r1 = np.stack([1-2*(y**2+z**2), 2*(x*y-s*z), 2*(x*z+s*y)], axis=1)
    r2 = np.stack([2*(x*y+s*z), 1-2*(x**2+z**2), 2*(y*z-s*x)], axis=1)
    r3 = np.stack([2*(x*z-s*y), 2*(y*z+s*x), 1-2*(x**2+y**2)], axis=1)
    return np.stack([r1, r2, r3], axis=1)

# 将位姿转换为矩阵
def pose_vec2mat(pvec, use_filler=True):
    """Convert quaternion vector represention to SE3 group"""
    t, q = pvec[np.newaxis, 0:3], pvec[np.newaxis, 3:7]
    R = quat2rotm(q)
    t = np.expand_dims(t, axis=-1)
    # 最终的转换矩阵
    P = np.concatenate([R, t], axis=2)
    if use_filler:
        filler = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 1, 4])
        P = np.concatenate([P, filler], axis=1)
    return P[0]

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
        # 获取索引数组
        inds = np.arange(num_frames)
        inds = inds[~np.equal(inds, keyframe_index)]
        
        inds = np.random.choice(inds, num_samples, replace=False)
        inds = [keyframe_index] + inds.tolist()
        # 读取图像
        images = []
        for i in inds:
            print("images:{}".format(data_blob['images'][i]))
            image = cv2.imread(data_blob['images'][i])
            image = cv2.resize(image, (640, 480))
            images.append(image)
        # 转换位姿信息
        poses = []
        for i in inds:
            pose_vec = data_blob['poses'][i]
            pose_mat = pose_vec2mat(pose_vec)
            poses.append(np.linalg.inv(pose_mat))
        # 转换图像和深度信息
        images = np.stack(images, axis=0).astype(np.uint8)
        poses = np.stack(poses, axis=0).astype(np.float32)
        # 获取深度信息
        depth_file = data_blob['depth']
        # 读取深度信息
        depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
        
        depth = (depth.astype(np.float32)) /factor
        filled = fill_depth(depth)
        
        K = data_blob['intrinsics']
        # 相机内参，转换为向量矩阵
        kvec = K.copy()

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
        datum_file = os.path.join(sequence_dir, 'pickle-TUM.pkl')
        # 如果序列化文件不存在就进行创建
        #if not os.path.isfile(datum_file):
            # 获取对齐之后的数据 
        images, depths, poses = get_TUM_data(sequence_dir)
        depth_intrinsics = intrinsics.copy()
        color_intrinsics = intrinsics.copy()
        # 写入序列化数据
        datum = images, depths, poses, color_intrinsics, depth_intrinsics
        #pickle.dump(datum, open(datum_file, 'wb'))
            
        #else:
        #    datum = pickle.load(open(datum_file, 'rb'))

        return datum


    def build_dataset_index(self, r=4, skip=12):
        self.dataset_index = []
        data_id = 0
        # 访问数据文件夹，并列举所有数据文件夹
        for scan in sorted(os.listdir(self.dataset_path)):
            print("scan:{}".format(scan))
            # 访问数据文件夹，构造对应的数据
            images, depths, poses, color_intrinsics, depth_intrinsics = self._load_scan(scan)

            for i in range(r, len(images)-r, skip):
                # some poses in scannet are nans
                if np.any(np.isnan(poses[i-r:i+r+1])):
                    continue
                # 加载网络数据参数
                training_example = {}
                training_example['depth'] = depths[i]
                training_example['images'] = images[i-r:i+r+1]
                training_example['poses'] = poses[i-r:i+r+1]
                training_example['intrinsics'] = depth_intrinsics
                training_example['n_frames'] = 2*r+1
                training_example['id'] = data_id

                self.dataset_index.append(training_example)
                data_id += 1

    #def test_set_iterator(self):

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