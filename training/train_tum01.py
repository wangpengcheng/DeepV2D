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

def main(args):

    cfg = config.cfg_from_file(args.cfg)
    # 
    log_dir = os.path.join('logs/tum', args.name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    checkpoint_dir = os.path.join('pytorch/tum', args.name)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    tmp_dir = os.path.join('tmp/tum', args.name)
    # 检查对应的文件夹是否存在
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    cfg.LOG_DIR = log_dir
    cfg.CHECKPOINT_DIR = checkpoint_dir
    cfg.TMP_DIR = tmp_dir
    # 创建数据集
    db = TUM_RGBD(cfg.INPUT.RESIZE, args.dataset_dir, r=args.r)

    solver = DeepV2DTrainer(cfg)
    ckpt = None
    # 进行训练
    solver.train(db, cfg, stage=2, restore_ckpt=args.restore, num_gpus=args.num_gpus)



if __name__ == '__main__':
   
    #设置value的显示长度为100，默认为50
    seed = 1234
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', help='name of your experiment')
    parser.add_argument('--dataset_dir', help='path to scannet directory')
    parser.add_argument('--cfg', default='cfgs/tum.yaml', help='path to yaml config file')
    parser.add_argument('--restore', help='checkpoint to restore')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus to use for training')
    parser.add_argument('--r', type=int, default=4, help='frame radius') # 帧半径
    args = parser.parse_args()

    main(args)
