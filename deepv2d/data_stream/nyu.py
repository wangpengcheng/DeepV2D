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

import threading, time
import sys
from utils.tum_associate import *


factor = 5000.0 # for the 16-bit PNG files 
# OR: factor = 1 # for the 32-bit float images in the ROS bag files
fx = 5.1885790117450188e+02
fy = 5.1946961112127485e+02
cx = 3.2558244941119034e+02
cy = 2.5373616633400465e+02
# 相机基本参数
intrinsics = np.array([fx, fy, cx, cy], dtype=np.float32)


def get_data(data_path, sum_data_file_name):
    """读取tum中的数据

    Args:
        data_path ([str]): 数据存在的路径 
    """
    data_file_path = os.path.join(data_path, sum_data_file_name)
    print("==== {} ======".format(data_file_path))
    # 不存在综合数据就进行创建
    # 在这里进行时间戳上的文件合并
    if not os.path.isfile(data_file_path):
        print("not data file {}".format(data_file_path))
    # 读取综合数据文件
    # 文件样例:
    # 1341841310.82 rgb/1341841310.821702.png 1341841310.82 depth/1341841310.822401.png 1341841310.81 -2.7366 0.1278 1.2413 0.7256 -0.5667 0.2209 -0.3217
    #
    image_names, depths_names, poses = get_data_from_sum_file(data_file_path)
    # 获取图像相对路径
    image_names = [data_path+'/'+i for i in image_names ]
    # 获取深度图像相对路径
    depth_names = [data_path+'/'+i for i in depths_names ]
    
    
    return image_names, depth_names, poses

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
    """
    Convert quaternion into rotation matrix 
    转换为捐助矩阵，主要是要是偏执计算
    """
    q /= np.sqrt(np.sum(q**2))
    #  旋转矩阵
    x, y, z, s = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # 计算三个方向的旋转
    r1 = np.stack([1-2*(y**2+z**2), 2*(x*y-s*z), 2*(x*z+s*y)], axis=1)
    r2 = np.stack([2*(x*y+s*z), 1-2*(x**2+z**2), 2*(y*z-s*x)], axis=1)
    r3 = np.stack([2*(x*z-s*y), 2*(y*z+s*x), 1-2*(x**2+y**2)], axis=1)
    return np.stack([r1, r2, r3], axis=1)

# 将位姿转换为矩阵
def pose_vec2mat(pvec, use_filler=True):
    """
    Convert quaternion vector represention to SE3 group
    将pose转换为se3李代数
    """
    # 提取位移和旋转
    t, q = pvec[np.newaxis, 0:3], pvec[np.newaxis, 3:7]
    R = quat2rotm(q)
    t = np.expand_dims(t, axis=-1)
    # 最终的转换矩阵
    P = np.concatenate([R, t], axis=2)
    if use_filler:
        filler = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 1, 4])
        P = np.concatenate([P, filler], axis=1)
    return P[0]

def build_data_map(data_blob_arr, width, height , thread_num =0):
    images_map = {}
    depths_map = {}
    
    # 读取图像
    for data in data_blob_arr:
        images_names = data['images']
        depth_name = data['depth']
        for image_name in images_names:
            #print("read image file:{}".format(image_name))
            image = cv2.imread(image_name)
            image = cv2.resize(image, (int(width), int(height)))
            images_map[image_name]= image

        # 读取深度信息
        depth = cv2.imread(depth_name, cv2.IMREAD_ANYDEPTH)
        depth = cv2.resize(depth, (int(width), int(height)))
        depth = (depth.astype(np.float32))/factor
        depths_map[depth_name] = depth

    print("thread_num:{} read depth and images file OK".format(thread_num))

    return images_map, depths_map

# MyThread.py线程类
class MyThread(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args
 
    def run(self):
        time.sleep(2)
        self.result = self.func(*self.args)
 
    def get_result(self):
        threading.Thread.join(self) # 等待线程执行完毕
        try:
            return self.result
        except Exception:
            return None

class NYU:
    """
    NYU数据集加载类，主要用来加载数据集中的数据
    主要用来进行数据的加载与查找
    """
    def __init__(self, resize, dataset_path, test=False, n_frames=5, r=2, scenes_file='data/nyu/train_scenes.txt', buffer_len = 100, load_thread_num = 4):
        """[summary]

        Args:
            resize ([type]): [description]
            dataset_path ([type]): [description]
            test (bool, optional): [description]. Defaults to False.
            n_frames (int, optional): [description]. Defaults to 5.
            r (int, optional): [description]. Defaults to 2.
            scenes_file: 数据集使用的场景文件夹
            buffer_len (int, optional): [description]. Defaults to 100.
            load_thread_num (int, optional): 缓存更新加载线程数. Defaults to 4.
        """
        self.dataset_path = dataset_path
        self.resize = resize
        self.n_frames = n_frames
        self.height = int(480*self.resize)
        self.width = int(640*self.resize)
        self.is_test = test
        self.load_thread_num = load_thread_num
        self.buffer_len = buffer_len
        self.images_map = {}
        self.depths_map = {} 
        self.scenes_list_file = scenes_file
        self.build_dataset_index(r=r, skip = 10)
        #self.check_files()

    def check_files(self):
        for i in range(len(self.dataset_index)):
            test = self.__getitem__(i)
        print("check ok")

    # 获取数据长度
    def __len__(self):
        return len(self.dataset_index)
    # 获取数据长度
    def shape(self):
        return [self.n_frames, self.height, self.width]
    # 获取数据
    def __getitem__(self, index):
        # if index+1 % self.buffer_len == 0:
        #     # 构建索引表
        #     self.flash_buffer(index)
        
        # 获取索引
        data_blob = self.dataset_index[index]
        # 获取关键帧长度
        num_frames = data_blob['n_frames']
        # 图片样例数量
        num_samples = self.n_frames - 1
        # 关键帧id
        frameid = data_blob['id']
        
        keyframe_index = num_frames // 2 # 选取中间帧作为关键帧
        # 创建关键帧,范围数组0~n-1 : 0,1,2,3,4
        inds = np.arange(num_frames)
        # 构建索引数组 True,  True, False,  True,  True -->  [0, 1, 3, 4]
        inds = inds[~np.equal(inds, keyframe_index)]
        # 对顺序进随机组合 [2, 4, 3, 1]
        #inds = np.random.choice(inds, num_samples, replace=False)
        # 将关键帧提取到开头 0,2,4,3
        inds = [keyframe_index] + inds.tolist()
        
        # 读取深度数据
          # 深度数据文件
        depth_file = data_blob['depth']
        # 进行图像读取
        depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
        depth = cv2.resize(depth, (int(self.width), int(self.height)))
        # 进行深度读取
        depth = (depth.astype(np.float32)) / factor
         # 深度平滑处理
        filled = fill_depth(depth)
        # 关键帧序列
        frameid = data_blob['id']
        # 转换为
        frameid = np.int32(frameid)
        # 读取图像
        images = []
        poses = []
        # 读取rgb
        for i in inds:
            image_file = data_blob['images'][i]
            img = cv2.imread(image_file)
            img = cv2.resize(img, (int(self.width), int(self.height)))
            images.append(img)
            pose_vec = data_blob['poses'][i]
            pose_mat = pose_vec2mat(pose_vec)
            poses.append(np.linalg.inv(pose_mat))
       
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

        #if not os.path.isfile(datum_file):
            # 获取对齐之后的数据 
        if self.is_test :
            sum_data_file_name = "rgb_depth_ground_test.txt"  # 注意这里测试数据的生成
        else:
            sum_data_file_name = "rgb_depth_ground.txt"
        # 获取相关的数据
        images, depths, poses = get_data(sequence_dir, sum_data_file_name)

        depth_intrinsics = intrinsics.copy()
        color_intrinsics = intrinsics.copy()
        # 写入序列化数据
        datum = images, depths, poses, color_intrinsics, depth_intrinsics

        return datum
    def get_dirs(self, file_name):
        """
        读取指定文件夹下的所有文件目录
        Args:
            dataset_path ([type]): [description]
            file_name ([type]): [description]
        """
        # 打开文件，按行读取
        reader = csv.reader(open(file_name))
        scenes = [x[0] for x in reader]
        return scenes

    def build_dataset_index(self, r=2, skip=12):
        self.dataset_index = []
        data_id = 0
       
        # 扫描场景名称
        scenes_names = self.get_dirs(self.scenes_list_file)
        scenes_names.sort()
        # 访问数据文件夹，并列举所有数据文件夹
        for scene in scenes_names:
            
            # 访问数据文件夹，构造对应的数据
            images, depths, poses, color_intrinsics, depth_intrinsics = self._load_scan(scene)
            # 注意这里的参数直接进行变换
            color_intrinsics = color_intrinsics*self.resize
            # 
            depth_intrinsics = depth_intrinsics*self.resize
            
            # 进行数据加载和 
            for i in range(r, len(images)-r, skip):
                # some poses in scannet are nans
                if np.any(np.isnan(poses[i-r:i+r+1])):
                    continue
                # 加载网络数据参数
                training_example = {}
                # 加载深度图像
                training_example['depth'] = depths[i]
                training_example['images'] = images[i-r:i+r+1]
                training_example['poses'] = poses[i-r:i+r+1]
                training_example['intrinsics'] = depth_intrinsics
                training_example['n_frames'] = 2*r+1
                training_example['id'] = data_id
                self.dataset_index.append(training_example)

                data_id += 1
    def flash_buffer(self, index):
        """
        刷新缓存数量
        Args:
            index ([type]): [description]
        """
        # 按照线程进行数据分割
        data_blob_arr = [self.dataset_index[i:i+self.load_thread_num] for i in range(index, index+self.buffer_len, self.load_thread_num)]
        image_map = {}
        depth_map = {}
        # 创建线程队列
        threads = []
        # 执行循环并行加载数据
        for i in range(self.load_thread_num):
            thread = MyThread(build_data_map, (data_blob_arr[i], self.width, self.height, i))
            thread.start()
            threads.append(thread)

        # 等待线程执行结束
        for thread in threads:
            temp_image_map, temp_depth_map = thread.get_result()
            image_map.update(temp_image_map)
            depth_map.update(temp_depth_map)
        
        self.images_map = image_map
        self.depths_map = depth_map


    #def test_set_iterator(self):
    

    def test_set_iterator(self):
        """
        测试数据迭代器,用来获取测试数据
        """
        
        # 遍历预加载的数据集
        for temp_data_blob in self.dataset_index:
            num_frames = temp_data_blob['n_frames']
            num_samples = self.n_frames - 1
            frameid = temp_data_blob['id']
            keyframe_index = num_frames//2
            # 图片索引
            inds = np.arange(num_frames)
            inds = inds[~np.equal(inds, keyframe_index)]
            inds = np.random.choice(inds, num_samples, replace=False)
            inds = [keyframe_index] + inds.tolist()
            # 添加图片数据
            images = []
            for i in inds:
                image = self.images[self.images_map[temp_data_blob['images'][i]]]
                images.append(image)

            # 转换图像
            images = np.stack(images, axis=0).astype(np.uint8)
            # 获取深度信息
            depth_file_name = temp_data_blob['depth']
            # 读取深度信息
            depth = self.depths[self.depths_map[depth_file_name]]
            # 获取位姿信息
            pose_vec = temp_data_blob['poses'][keyframe_index]
            pose = pose_vec2mat(pose_vec)
            K = temp_data_blob['intrinsics']
            # 相机内参，转换为向量矩阵
            kvec = K.copy()
            # 转换深度信息
            # depth = depth[...,None]
            data_blob = {
                    'images': images,
                    'depth': depth,
                    'pose': pose,
                    'intrinsics': intrinsics,
            }
            yield data_blob


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