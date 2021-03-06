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
import threading, time
from utils.tum_associate import *

fx = 517.3
fy = 516.5
cx = 318.6
cy = 255.3
factor = 1000.0 # for the 16-bit PNG files 
# OR: factor = 1 # for the 32-bit float images in the ROS bag files

intrinsics = np.array([fx, fy, cx, cy],dtype=np.float32)

def list_split(items, n):
    """
    数组切分
    Args:
        items ([type]): [description]
        n ([type]): [description]

    Returns:
        [type]: [description]
    """
    return [items[i:i+n] for i in range(0, len(items), n)]

def build_fast_data_map(
            data_blob_arr, 
            img_buffer, 
            depth_buffer,
            img_map,
            depth_map,
            width, 
            height, 
            thread_num=0, 
            buffer_len = 100
            ):
        # 开始坐标
        start_index = buffer_len*thread_num

        i = 0
        # 读取图像
        for data in data_blob_arr:
            images_names = data['images']
            depth_name = data['depth']
            img_len = len(images_names)
            j = 0
            for image_name in images_names:
                #print("read image file:{}".format(image_name))
                image = cv2.imread(image_name)
                image = cv2.resize(image, (int(width), int(height)))
                # 计算索引
                img_index = (start_index+i)*img_len + j
                # 更新索引
                img_buffer[img_index] = image
                img_map[image_name] = img_index
                j = j+1
            # 读取深度信息
            depth = cv2.imread(depth_name, cv2.IMREAD_ANYDEPTH)
            depth = cv2.resize(depth, (int(width), int(height)))
            depth = (depth.astype(np.float32))/factor
            # if np.any( depth == 0 ):
            #     print("{} is empty".format(depth_name))
            depth_index = start_index+i

            depth_buffer[depth_index] = depth
            depth_map[depth_name] = depth_index
            i = i+1
        print("thread_num:{} read depth and images file OK".format(thread_num))

# MyThread.py 线程类
class MyThread(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args
 
    def run(self):
        self.result = self.func(*self.args)
 
    def get_result(self):
        print("===== Thread Join ======")
        threading.Thread.join(self) # 等待线程执行完毕
        try:
            return self.result
        except Exception:
            print("Thread Error!!!!")   
            return None


def get_TUM_data(data_path, sum_data_file_name):
    """读取tum中的数据

    Args:
        data_path ([str]): 数据存在的路径 
    """
    data_file_path = os.path.join(data_path, sum_data_file_name)
    print("==== {} ======".format(data_file_path))
    # 不存在综合数据就进行创建
    # 在这里进行时间戳上的文件合并
    if not os.path.isfile(data_file_path):
        print("File Not exit!!!!!!")
    # 读取综合数据文件
    # 文件样例:
    # 1341841310.82 rgb/1341841310.821702.png 1341841310.82 depth/1341841310.822401.png 1341841310.81 -2.7366 0.1278 1.2413 0.7256 -0.5667 0.2209 -0.3217
    #
    image_names, depths_names, poses = get_data_from_sum_file(data_file_path)

    image_names = [data_path+'/'+i for i in image_names ]
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
    # 获取宽和高
    x, y = np.meshgrid(np.arange(depth.shape[1]).astype("float32"),
                       np.arange(depth.shape[0]).astype("float32"))
    # 获取相关坐标
    xx = x[depth > 0]
    yy = y[depth > 0]
    zz = depth[depth > 0]
    if (xx.size == 0 ) or (yy.size == 0) or (zz.size == 0):
        return depth
    
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
    # 位移矩阵
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
    def __init__(self, resize, dataset_path, test=False, n_frames=5, r=2):
        self.dataset_path = dataset_path
        self.resize = resize
        self.n_frames = n_frames
        self.height = int(480*self.resize)
        self.width = int(640*self.resize)
        self.is_test = test
        # 多线程加载数据数目
        self.load_thread_num = 16
        # 读取图像
        self.images = []
        self.images_map = {}
        self.img_index= 0
        # 读取深度
        self.depths = []
        self.depths_map={}
        self.depth_index = 0
        
        self.build_dataset_index(r=r)
        self.build_fast_map()
        
    # 获取数据长度
    def __len__(self):
        return len(self.dataset_index)
    # 获取数据长度
    def shape(self):
        return [self.n_frames, self.height, self.width]
    # 获取数据
    def __getitem__(self, index):
        try:
            # 获取索引
            data_blob = self.dataset_index[index]
            num_frames = data_blob['n_frames']
            # 图片样例数量
            num_samples = self.n_frames - 1

            frameid = data_blob['id']
            
            keyframe_index = num_frames // 2 # 选取中间帧作为关键帧
            # 创建关键帧
            inds = np.arange(num_frames)
            inds = inds[~np.equal(inds, keyframe_index)]
            
            inds = np.random.choice(inds, num_samples, replace=False)
            # 将关键帧提取到开头
            inds = [keyframe_index] + inds.tolist()
            # 读取图像
            images = []
            for i in inds:
                image = self.images[self.images_map[data_blob['images'][i]]]
                images.append(image)
            # 转换位姿信息
            poses = []
            for i in inds:
                pose_vec = data_blob['poses'][i]
                pose_mat = pose_vec2mat(pose_vec)
                #poses.append(np.linalg.inv(pose_mat))
                poses.append(pose_mat)
            # 转换图像和深度信息
            images = np.stack(images, axis=0).astype(np.uint8)
            poses = np.stack(poses, axis=0).astype(np.float32)
            # 获取深度信息
            depth_file = data_blob['depth']
            # 读取深度信息
            depth = self.depths[self.depths_map[depth_file]]
            myfilled = fill_depth(depth)
            K = data_blob['intrinsics']
            # 相机内参，转换为向量矩阵
            kvec = K.copy()

            depth = depth[..., None]
            return images, poses, depth, myfilled, myfilled, kvec, frameid
        except BaseException:
            print(" error ", index, depth_file)
        


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

        # 如果序列化文件不存在就进行创建
        #if not os.path.isfile(datum_file):
            # 获取对齐之后的数据 
        if self.is_test :
            sum_data_file_name = "test.txt" # 测试文件
        else:
            sum_data_file_name = "train.txt" # 训练文件
        images, depths, poses = get_TUM_data(sequence_dir, sum_data_file_name)
        # 获取相机内参
        #加载信息
        my_intrinsics = np.loadtxt(os.path.join(sequence_dir, 'intrinsics.txt'))
        depth_intrinsics = my_intrinsics.copy()
        color_intrinsics = my_intrinsics.copy()
        # 写入序列化数据
        datum = images, depths, poses, color_intrinsics, depth_intrinsics

        return datum


    def build_dataset_index(self, r=2, skip=5):
        self.dataset_index = []
        data_id = 0
        # 访问数据文件夹，并列举所有数据文件夹
        for scan in sorted(os.listdir(self.dataset_path)):
            # 访问数据文件夹，构造对应的数据
            images, depths, poses, color_intrinsics, depth_intrinsics = self._load_scan(scan)
            # 注意这里的参数直接进行变换
            #color_intrinsics = color_intrinsics*self.resize 
            depth_intrinsics = depth_intrinsics*self.resize
            # 构建索引表
            #self.build_data_map(images, depths, poses)
            # 加载数据
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

    
    def build_fast_map(self):
        # 进行内存空间分配
        self.images = [0]*(int(len(self.dataset_index)*self.n_frames)+1)
        self.depths = [0]*(int(len(self.dataset_index))+1)

        # 进行数据分割
        data_blob_arr = list_split(self.dataset_index, len(self.dataset_index)//(self.load_thread_num))
        # 创建线程队列
        threads = []
        buffer_len = len(data_blob_arr[0])
        # 执行循环并行加载数据
        for i in range(len(data_blob_arr)):
            thread = MyThread(build_fast_data_map, (data_blob_arr[i], self.images, self.depths, self.images_map,self.depths_map,  self.width, self.height, i, buffer_len))
            thread.start()
            threads.append(thread)

        # 等待线程执行结束
        for thread in threads:
            # 获取最终map
            thread.get_result()

        print(len(self.images))
        print(len(self.depths))
        print(len(self.images_map))
        print(len(self.depths_map))

    #def test_set_iterator(self):
    def build_data_map(self, images_names, depths_names, poses):

        for image_name in images_names:
            #print("read image file:{}".format(image_name))
            image = cv2.imread(image_name)
            image = cv2.resize(image, (self.width, self.height))
            self.images.append(image)
            self.images_map[image_name]= self.img_index
            self.img_index = self.img_index +1
        print("read image file OK")
      
        for depth_name in depths_names:
            # 读取深度信息
            depth = cv2.imread(depth_name, cv2.IMREAD_ANYDEPTH)
            depth = cv2.resize(depth, (int(self.width), int(self.height)))
            depth = (depth.astype(np.float32))/factor
            self.depths.append(depth)
            self.depths_map[depth_name]= self.depth_index
            self.depth_index = self.depth_index+1
        print("read depth file OK")

    def test_set_iterator(self):
        """
        测试数据迭代器
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
            poses = []
            for i in inds:
                pose_vec = temp_data_blob['poses'][i]
                pose_mat = pose_vec2mat(pose_vec)
                #poses.append(np.linalg.inv(pose_mat))
                poses.append(pose_mat)
            K = temp_data_blob['intrinsics']
            poses = np.stack(poses, axis=0).astype(np.float32)
            # 相机内参，转换为向量矩阵
            kvec = K.copy()
            # 转换深度信息
            # depth = depth[...,None]
            data_blob = {
                    'images': images,
                    'depth': depth,
                    'poses': poses,
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