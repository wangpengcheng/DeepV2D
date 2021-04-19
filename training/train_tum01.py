# -*- coding: utf-8 -*-
import sys
sys.path.append('deepv2d')
import torch
import numpy as np
import matplotlib.pyplot as plt

import cv2
import os
import time
import random
import argparse

from core import config
from my_trainer import DeepV2DTrainer

from data_stream.tum import TUM_RGBD
from utils.my_utils import *

def main(args):

    cfg = config.cfg_from_file(args.cfg)
    #输出日志文件夹
    log_dir = os.path.join(cfg.STORE.LOG_DIR, cfg.STORE.MODLE_NAME)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    #输出文件夹
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

    # 创建数据集
    db = TUM_RGBD(cfg.INPUT.RESIZE, args.dataset_dir, r=args.r)

    solver = DeepV2DTrainer(cfg)
    ckpt = None
    # 进行训练
    solver.train(db, cfg, stage=2, restore_ckpt=restore_ckpt_path, num_gpus=args.num_gpus)



if __name__ == '__main__':
   
    # 设置value的显示长度为100，默认为50
    seed = 1234
    set_random_seed(seed)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', help='name of your experiment')
    parser.add_argument('--dataset_dir', help='path to scannet directory')
    parser.add_argument('--cfg', default='cfgs/tum.yaml', help='path to yaml config file')
    parser.add_argument('--restore', help='checkpoint to restore')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus to use for training')
    parser.add_argument('--r', type=int, default=2, help='frame radius') # 帧半径
    args = parser.parse_args()

    main(args)
