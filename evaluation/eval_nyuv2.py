import sys
sys.path.append('deepv2d')

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from data_stream.nyu import NYU
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
    """
    将相机内参写入参数
    Args:
        images ([type]): 输入图像
        intrinsics ([type]): 相机内参
        test_id ([type]): [description]
    """
    dest = os.path.join("tum/%06d" % test_id)

    if not os.path.isdir(dest):
        os.makedirs(dest)

    for i, img in enumerate(images):
        cv2.imwrite(os.path.join(dest, '%d.png'%i), img)

    np.savetxt(os.path.join(dest, 'intrinsics.txt'), intrinsics)



def make_predictions(args):
    """
    进行初始化够造函数
    Args:
        args ([type]): [description]

    Returns:
        [type]: [description]
    """
    # 设置随机数种子

    # 读取参数
    cfg = config.cfg_from_file(args.cfg)

    deepv2d = DeepV2D(
        cfg, 
        args.model, 
        use_fcrn=False, 
        mode=args.mode
        )
    # 进行初始化
    # init_op = tf.group(tf.global_variables_initializer(),
    #         tf.local_variables_initializer())

    # 设置运行环境
    set_gpus(cfg)
    # 开启运行
    with tf.Session() as sess:
        #sess.run(init_op)
        # 设置session
        deepv2d.set_session(sess)
        # 设置预测模型
        depth_predictions, pose_predictions = [], []
        depth_groundtruth, pose_groundtruth = [], []
        # 创建数据集
        test_sence_file = 'data/nyu/test_scenes.txt'
        # 构建数据加载器
        db = NYU(cfg.INPUT.RESIZE, args.dataset_dir, test_sence_file, test=False, n_frames=5, r=2, skip1 = 15)
        #提取数据集
        for test_id, test_blob in enumerate(db.test_set_iterator()):
            # 获取图像和相机位姿
            images, intrinsics, poses = test_blob['images'], test_blob['intrinsics'], test_blob['poses']
            # 进行推理
            depth_pred  = deepv2d.inference(images, poses, intrinsics)
            # 进行预测
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
    """ nyu evaluations """
    
    crop = [20//2, 459//2, 24//2, 615//2] # eigen crop
    gt_list = []
    pr_list = []

    num_test = len(predictions)
    # 进行数据遍历
    for i in range(num_test):
        depth_gt = groundtruth[i]
        depth_pr = predictions[i]

        # crop and resize
        depth_pr = cv2.resize(depth_pr, (320, 240))
        depth_pr = depth_pr[crop[0]:crop[1], crop[2]:crop[3]]
        depth_gt = depth_gt[crop[0]:crop[1], crop[2]:crop[3]]

        # scale predicted depth to match gt
        scalor = eval_utils.compute_scaling_factor(depth_gt, depth_pr, min_depth=0.8, max_depth=10.0)
        depth_pr = scalor * depth_pr

        gt_list.append(depth_gt)
        pr_list.append(depth_pr)

    depth_results = eval_utils.compute_depth_errors(gt_list, pr_list)
    print(("{:>10}, "*len(depth_results)).format(*depth_results.keys()))
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