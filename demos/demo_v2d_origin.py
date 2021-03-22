# -*- coding:UTF-8 -*-
import sys
sys.path.append('deepv2d')
sys.path.append('evaluation')
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import argparse
import cv2
import os
import time
import glob
import random
from data_stream.tum import *
from utils.count import * 
from deepv2d import vis
from core import config
from deepv2d import DeepV2D
from utils.my_utils import set_gpus
import eval_utils 

# fx = 535.4
# fy = 539.2
# cx = 320.1
# cy = 247.6
fx = 6.359763793945312500e+02
fy = 5.777985229492187500e+02
cx = 5.292523803710937500e+02
xy = 2.780844116210937500e+01
factor = 5000.0 # for the 16-bit PNG files 
# OR: factor = 1 # for the 32-bit float images in the ROS bag files
intrinsics = np.array([fx, fy, cx, cy],dtype=np.float32)

def load_test_sequence(path, n_frames=-1):
    """ loads images and intrinsics from demo folder """
    images = []
    #加载图像
    for imfile in sorted(glob.glob(os.path.join(path, "*.png"))):
        img = cv2.imread(imfile)
        images.append(img)

    inds = np.arange(1, len(images))
    # 随机选择n张图片作为关键帧
    if n_frames > 0:
        inds = np.random.choice(inds, n_frames, replace=False)
    #选取第一帧为关键帧
    inds = [0] + inds.tolist()
    print(inds)
    #获取所有图像
    images = [images[i] for i in inds]
    #图像转换为float 32位
    images = np.stack(images).astype(np.float32)
    #加载信息
    intrinsics = np.loadtxt(os.path.join(path, 'intrinsics.txt'))
    # 返回图像信息，相机内参
    return images, intrinsics

def load_sorted_test_sequence(data_path, inference_file_name, scale):

    image_names, depths_names, pre_poses = get_data_from_sum_file(inference_file_name)
    
    # 文件参数
    image_names = [data_path+'/'+i for i in image_names ]
    depths_names = [data_path+'/'+i for i in depths_names ]
    images= []
    # 加载所有图片信息
    for image_name in image_names:
        image = cv2.imread(image_name)
        image = cv2.resize(image, (int(640*scale), int(480*scale)))
        images.append(image)
    
    print("load rgb image: OK")
    poses = []
    for pre_pose in pre_poses:
        pose_mat = pose_vec2mat(pre_pose)
        # 注意这里对矩阵进行了一次求逆矩阵运算
        poses.append(np.linalg.inv(pose_mat))
    depths = []
    for depth_name in depths_names:
        depth = cv2.imread(depth_name, cv2.IMREAD_ANYDEPTH)
        depth = cv2.resize(depth, (int(640*scale), int(480*scale)))
        depth = (depth.astype(np.float32))/factor
        depths.append(depth)
    print("load rgb depths: OK")
    #相机内参
    my_intrinsics = np.loadtxt(os.path.join(data_path, 'intrinsics.txt'))
    # 注意这里的相机内参缩放
    my_intrinsics = my_intrinsics*scale 
    
    return images, poses, depths, my_intrinsics

def main(args):

    if args.cfg is None:
        if 'nyu' in args.model:
            args.cfg = 'cfgs/nyu.yaml'
        elif 'scannet' in args.model:
            args.cfg = 'cfgs/scannet.yaml'
        elif 'kitti' in args.model:
            args.cfg = 'cfgs/kitti.yaml'
        elif 'tum' in args.model:
            args.cfg = 'cfgs/tum.yaml'
        else:
            args.cfg = 'cfgs/nyu.yaml'
        
    cfg = config.cfg_from_file(args.cfg)
    is_calibrated = not args.uncalibrated
    is_pose = args.use_pose
    # 获取范围内的帧的长度
    frames_len = args.n_frames
    #build the DeepV2D graph
    # 构建深度推理图
    deepv2d = DeepV2D(cfg, args.model, use_fcrn=args.fcrn, is_calibrated=is_calibrated, mode=args.mode)
    
    set_gpus(cfg)
    
    with tf.Session() as sess:

        deepv2d.set_session(sess)
        #summary_writer = tf.summary.FileWriter('./log/', sess.graph)
        #加载图像和相机位姿初始值
        #call deepv2d on a video sequence
        # 加载测试数据集
        if not is_pose:
            # 注意这里保证其它接口
            images, intrinsics = load_test_sequence(args.sequence)
        # 进行图参数量的统计
        if 0:
            graph =tf.get_default_graph()
            stats_graph(graph)
        # 根据相机是否标定，来执行函数
        if is_pose:
            # 进行参数加载
            images ,poses, depths_gt, intrinsics = load_sorted_test_sequence(args.sequence, args.inference_file_name, cfg.INPUT.RESIZE)
            # 根据数据进行迭代，根据前面n帧的内容，推断最后帧的内容,注意这里推理的是中间关键帧的内容
            iter_number = int(len(images)/frames_len)
            time_sum =0.0
            # 遍历进行
            for i in range(0, iter_number):
                temp_images = images[i*frames_len:(i+1)*frames_len]
                temp_poses = poses[i*frames_len:(i+1)*frames_len]
                temp_intrinsics = intrinsics.copy()
                temp_images = np.stack(temp_images, axis=0).astype(np.uint8)
                temp_poses = np.stack(temp_poses, axis=0).astype(np.float32)
                # print("pose",temp_poses[0])
                # 计算时间
                time_start=time.time()
                # 进行推理
                depths = deepv2d.inference(temp_images, temp_poses, temp_intrinsics)
                time_end=time.time()
                print('time cost',time_end-time_start,'s')
                if i != 0:
                    time_sum = time_sum + (time_end-time_start)
                # 关键帧
                key_frame_depth = depths[0]
                # 关键rgb帧
                key_frame_image = temp_images[0]
                # 关键深度帧
                depth_gt = depths_gt[i*frames_len]
                # 计算深度缩放
                scalor = eval_utils.compute_scaling_factor(depth_gt, key_frame_depth, min_depth=0.2, max_depth=8.0)
                key_frame_depth =  scalor * key_frame_depth
                # 对深度图像进行平滑处理
                # key_frame_depth = cv2.medianBlur(key_frame_depth,5)
                image_depth = vis.create_image_depth_figure(key_frame_image, key_frame_depth)
                # 创建结果文件夹
                result_out_dir = "{}/{}".format(args.sequence, "inference_result")
                # 检测路径文件夹
                if not os.path.exists(result_out_dir):
                    os.makedirs(result_out_dir)
                # 写入图片
                cv2.imwrite("{}/{}.png".format(result_out_dir, i), image_depth)
                print("wirte image:{}/{}.png".format(result_out_dir,i))

            print("{} images,totle time: {} s, avg time: {} s".format(iter_number-1,time_sum,time_sum/(iter_number-1)))
        elif is_calibrated:
            depths, poses = deepv2d(images, intrinsics, viz=True, iters=args.n_iters)
        else:
            depths, poses = deepv2d(images, viz=True, iters=args.n_iters)
        
       


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='config file used to train the model')
    parser.add_argument('--n_frames',type=int,default=5,help="key frames number")
    parser.add_argument('--inference_file_name',default='inference_result/test.txt',help='inference file name')
    parser.add_argument('--model', default='models/nyu.ckpt', help='path to model checkpoint')
    parser.add_argument('--use_pose',action='store_true',help='use pose data')
    parser.add_argument('--mode', default='keyframe', help='keyframe or global pose optimization')
    parser.add_argument('--fcrn', action="store_true", help='use fcrn for initialization')
    parser.add_argument('--n_iters', type=int, default=5, help='number of iterations to use')
    parser.add_argument('--uncalibrated', action="store_true", help='use fcrn for initialization')
    parser.add_argument('--sequence', help='path to sequence folder')
    args = parser.parse_args()

    main(args)
