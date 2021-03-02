import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import cv2
import torch.optim as optim

from geometry.transformation import *

from modules.depth_module import DepthModule



MOTION_LR_FRACTION = 0.1

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class DeepV2DTrainer(object):
    """
    网络训练基础类型类
    Args:
        object ([type]): [description]
    """
    def __init__(self, cfg):
        self.cfg = cfg

    def train(self, data_source, cfg, stage=1, ckpt=None, restore_ckpt=None, num_gpus=1):
        """主要的训练函数

        Args:
            data_source ([type]): [description]
            cfg ([type]): [description]
            stage (int, optional): [description]. Defaults to 1.
            ckpt ([type], optional): [description]. Defaults to None.
            restore_ckpt ([type], optional): [description]. Defaults to None.
            num_gpus (int, optional): [description]. Defaults to 1.

        Returns:
            [type]: [description]
        """
        is_use_gpu = False
        # 获取深度步长
        batch_size = num_gpus * cfg.TRAIN.BATCH[stage-1]
        # 获取最大步长
        max_steps = cfg.TRAIN.ITERS[stage-1]
        # 设置GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.TRAIN.USE_GPU
        # 设置存储频率
        SUMMARY_FREQ = 10
        # 设置日志频率
        LOG_FREQ = 100
        # 设置checkpoint中间输出频率
        CHECKPOINT_FREQ = 5000
        # 设置最大步长
        self.training_steps = max_steps
        # 开始加载数据模型
        print ("batch size: %d \t max steps: %d"%(batch_size, max_steps))
        # 初始化网络
        deepModel = DepthModule(cfg)
         # 加载模型
        if restore_ckpt is not None:
            # 加载模型
            deepModel.load_state_dict(torch.load(restore_ckpt))
        if is_use_gpu:
            if torch.cuda.device_count() > 1:
                deepModel = nn.DataParallel(deepModel)
            deepModel = deepModel.cuda()


        # 定义损失函数
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.SGD(deepModel.parameters(), lr=cfg.TRAIN.LR, momentum=0.9)
        running_loss = 0.0

        # 设置训练数据集
        trainloader = torch.utils.data.DataLoader(data_source, batch_size=batch_size, shuffle=False, num_workers=2)
       

        for training_step in range(max_steps):
            # 开始加载数据
            for i, data in enumerate(trainloader,0):
                images_batch, poses_batch, gt_batch, filled_batch, pred_batch, intrinsics_batch, frame_id= data
                # 进行数据预处理
                Ts = VideoSE3Transformation(matrix=poses_batch)
                images = images_batch.permute(0, 1, 4, 2, 3)
                

                # set optimizer buffer to 0
                optimizer.zero_grad()
                # 前向计算
                outputs = deepModel(Ts, images, intrinsics_batch)
                # 计算loss值
                loss = loss_function(outputs, gt_batch)
                # loss backward
                loss.backward()
                # update parameters using optimizer
                optimizer.step()
                # 计算loss
                running_loss += loss.item()

            # 输出loss值
            if training_step % LOG_FREQ == 0:
                print('[step=%5d] loss: %.9f'%(step, running_loss / LOG_FREQ))
                running_loss = 0.0
            # 输出模型
            if training_step % CHECKPOINT_FREQ == 0 or training_step+1 == max_steps:
                # 模型名称
                save_file = os.path.join(cfg.CHECKPOINT_DIR, cfg.TRAIN.MODULE_NAME)
                # 模型名字
                torch.save(deepModel.cpu().module.state_dict(), save_file)




