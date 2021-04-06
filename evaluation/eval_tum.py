import sys
sys.path.append('deepv2d')

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import cv2
import os
import time
import argparse
import glob

import vis
from utils.count import *
from core import config
from data_stream.tum import TUM_RGBD
from deepv2d_clear import DeepV2D
from utils.my_utils import set_gpus
import eval_utils


def write_to_folder(images, intrinsics, test_id):
    dest = os.path.join("tum/%06d" % test_id)

    if not os.path.isdir(dest):
        os.makedirs(dest)

    for i, img in enumerate(images):
        cv2.imwrite(os.path.join(dest, '%d.png'%i), img)

    np.savetxt(os.path.join(dest, 'intrinsics.txt'), intrinsics)



def make_predictions(args):

    cfg = config.cfg_from_file(args.cfg)
    # 模型初始化
    deepv2d = DeepV2D(cfg, args.model,  mode=args.mode)
    # 进行初始化
    # init_op = tf.group(tf.global_variables_initializer(),
    #         tf.local_variables_initializer())
    set_gpus(cfg)
    # 开启运行
    with tf.Session() as sess:
        #sess.run(init_op)
        # 设置session
        deepv2d.set_session(sess)
        # 进行参数量的统计
        # graph =tf.get_default_graph()
        # stats_graph(graph)
        # 设置预测模型
        depth_predictions, pose_predictions = [], []
        depth_groundtruth, pose_groundtruth = [], []
        # 构建数据加载器
        db = TUM_RGBD(cfg.INPUT.RESIZE, args.dataset_dir, test=True, n_frames=5, r=2)
        #dl = DataLayer(db, batch_size=1)
        time_sum =0.0
        i = 0
        #提取数据集
        for test_id, test_blob in enumerate(db.test_set_iterator()):
            # 获取图像和相机位姿以及真实值
            images, intrinsics, gt, poses = test_blob['images'], test_blob['intrinsics'], test_blob['depth'], test_blob['poses']
            # 计算时间
            time_start = time.time()
            
            # 进行推理
            depth_pred = deepv2d.inference(images, poses, intrinsics)
            # 结束时间
            time_end = time.time()
            print('time cost', time_end - time_start,' s')
            if i != 0:
                time_sum = time_sum + (time_end-time_start)
            i = i + 1
            # use keyframe depth for evaluation
            depth_predictions.append(depth_pred[0])
            # 添加真实数据
            depth_groundtruth.append(test_blob['depth'])
        print("{} images,totle time: {} s, avg time: {} s".format(i-1, time_sum, time_sum/(i-1)))
    # 预测深度与位姿
    predictions = depth_predictions
    # 真实值
    groundtruth = depth_groundtruth
    # 返回预测值与真实值
    return groundtruth, predictions


def evaluate(groundtruth, predictions):
    crop = [20//2, 459//2, 24//2, 610//2] # eigen crop
    depth_results = {}
    # 真实值
    depth_groundtruth  = groundtruth
    # 预测值
    depth_predictions  = predictions

    # 
    num_test = len(depth_groundtruth)
    for i in range(num_test):
        # 真实数据
        depth_gt = groundtruth[i]
        # 预测数据
        depth_pr = predictions[i]
        depth_pr = cv2.resize(depth_pr, (640//2, 480//2))
        depth_pr = depth_pr[crop[0]:crop[1], crop[2]:crop[3]]
        depth_gt = depth_gt[crop[0]:crop[1], crop[2]:crop[3]]
        # match scales using median

        scalor = eval_utils.compute_scaling_factor(depth_gt, depth_pr)
        
        depth_pr = scalor * depth_pr

        # 计算深度误差
        depth_metrics = eval_utils.compute_depth_errors(depth_gt, depth_pr)
        # 将关键帧设置为空
        if i == 0:
            for dkey in depth_metrics:
                depth_results[dkey] = []

        for dkey in depth_metrics:
            depth_results[dkey].append(depth_metrics[dkey])
    # 对所有的项目求均值
    for dkey in depth_results:
        depth_results[dkey] = np.mean(depth_results[dkey])

    print(("{:>1}, "*len(depth_results)).format(*depth_results.keys()))
    print(("{:10.4f}, "*len(depth_results)).format(*depth_results.values()))



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='cfgs/tum.yaml', help='config file used to train the model')
    parser.add_argument('--model', default='models/tum.ckpt', help='path to model checkpoint')
    parser.add_argument('--dataset_dir', help='path to tmu test dataset')
    parser.add_argument('--mode', default='keyframe', help='config file used to train the model')
    parser.add_argument('--fcrn', action="store_true", help='use single image depth initializiation')
    parser.add_argument('--n_iters', type=int, default=8, help='number of video frames to use for reconstruction')
    parser.add_argument('--viz', action="store_true", help='display depth maps during inference')

    args = parser.parse_args()
    # 进行预测推理，获取
    groundtruth, predictions = make_predictions(args)
    #evaluate(groundtruth, predictions)