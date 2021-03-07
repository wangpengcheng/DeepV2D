import sys
sys.path.append('deepv2d')

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import cv2
import os
import time
import random
import argparse

from core import config
from trainer import DeepV2DTrainer

from data_stream.nyu import NYU

def main(args):

    cfg = config.cfg_from_file(args.cfg)
    # 设置日志级别
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = cfg.STORE.LOG_LEVEL
    # 输出日志文件夹
    log_dir = os.path.join(cfg.STORE.LOG_DIR, cfg.STORE.MODLE_NAME)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # 输出文件夹
    checkpoint_dir = os.path.join(cfg.STORE.CHECKPOINT_DIR, cfg.STORE.MODLE_NAME)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # 临时文件夹名称
    tmp_dir = os.path.join(cfg.STORE.TMP_DIR, cfg.STORE.MODLE_NAME)
    # 检查对应的文件夹是否存在
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    # 设置日志文件系统
    cfg.LOG_DIR = log_dir
    # 模型文件存储系统
    cfg.CHECKPOINT_DIR = checkpoint_dir
    cfg.TMP_DIR = tmp_dir
    # 已存在模型位置
    restore_ckpt_path = cfg.STORE.RESRORE_PATH


    solver = DeepV2DTrainer(cfg)
    ckpt = None
    # 注意这里直接使用tfrecords进行训练
    if args.restore is not None:
        solver.train(args.tfrecords, cfg, stage=2, restore_ckpt=args.restore, num_gpus=args.num_gpus)

    else:
        for stage in [1, 2]:
            ckpt = solver.train(args.tfrecords, cfg, stage=stage, ckpt=ckpt, num_gpus=args.num_gpus)
            tf.reset_default_graph()


if __name__ == '__main__':

    seed = 1234
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', help='name of your experiment')
    parser.add_argument('--cfg', default='cfgs/nyu.yaml', help='path to yaml config file')
    parser.add_argument('--tfrecords', default='datasets/nyu_train.tfrecords', help='path to tfrecords training file')
    parser.add_argument('--dataset_dir', default='data/nyu2', help='path to data training file')
    parser.add_argument('--restore',  help='use restore checkpoint')
    parser.add_argument('--num_gpus',  type=int, default=1, help='number of gpus to use')
    parser.add_argument('--r', type=int, default=2, help='frame radius') # 帧半径
    args = parser.parse_args()

    main(args)