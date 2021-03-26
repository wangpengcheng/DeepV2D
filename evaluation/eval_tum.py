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

from core import config
from data_stream.tum import TUM_RGBD
from deepv2d import DeepV2D
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
    deepv2d = DeepV2D(cfg, args.model, use_fcrn=False, mode=args.mode)
    # 进行初始化
    # init_op = tf.group(tf.global_variables_initializer(),
    #         tf.local_variables_initializer())
    set_gpus(cfg)
    # 开启运行
    with tf.Session() as sess:
        #sess.run(init_op)
        # 设置session
        deepv2d.set_session(sess)
        # 设置预测模型
        depth_predictions, pose_predictions = [], []
        depth_groundtruth, pose_groundtruth = [], []
        # 构建数据加载器
        db = TUM_RGBD(cfg.INPUT.RESIZE, args.dataset_dir, test=True, n_frames=5, r=2)
        #提取数据集
        for test_id, test_blob in enumerate(db.test_set_iterator()):
            # 获取图像和相机位姿
            images, intrinsics, poses = test_blob['images'], test_blob['intrinsics'], test_blob['poses']
            # 进行推理
            depth_pred  = deepv2d.inference(images, poses, intrinsics)

            # use keyframe depth for evaluation
            depth_predictions.append(depth_pred[0])
            
            # 添加真实数据
            depth_groundtruth.append(test_blob['depth'])

    # 预测深度与位姿
    predictions = depth_predictions
    # 真实值
    groundtruth = depth_groundtruth
    # 返回预测值与真实值
    return groundtruth, predictions


def evaluate(groundtruth, predictions):

    depth_results = {}
    # 真实值
    depth_groundtruth  = groundtruth
    # 预测值
    depth_predictions  = predictions
    # 
    num_test = len(depth_groundtruth)
    for i in range(num_test):
        # match scales using median
        scalor = eval_utils.compute_scaling_factor(depth_groundtruth[i], depth_predictions[i])
        depth_predictions[i] = scalor * depth_predictions[i]
        # 计算深度误差
        depth_metrics = eval_utils.compute_depth_errors(depth_groundtruth[i], depth_predictions[i])
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
    evaluate(groundtruth, predictions)