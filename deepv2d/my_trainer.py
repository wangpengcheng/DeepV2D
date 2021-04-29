import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import cv2
import torch.optim as optim
import time
#from geometry.transformation import *
from tensorboardX import SummaryWriter

from modules.depth_module import DepthModule
from utils.my_utils import *

# from torchsummary import summary
# from geometry.transformation import *

# from modules.depth_module import DepthModule
from modules.my_loss import MyLoss


class data_prefetcher():
    def __init__(self, cfg, loader):
        # 加载配置参数
        self.cfg = cfg
        # 加载数据
        self.origin_loader = list(loader)
        # 计算加载器长度
        self.len = len(self.origin_loader)
        print(self.len)
        # 当前索引
        self.currt_index = 0
        self.id = 0
        # 运算流
        self.stream = torch.cuda.Stream()
        # 进行预加载
        self.preload()
    def reset(self):
        """
        重新设置数据集
        """
        self.currt_index = 0
        self.preload()


    def preload(self):
        if self.currt_index < self.len:
            images_batch, poses_batch, gt_batch, filled_batch, pred_batch, intrinsics_batch, frame_id = self.origin_loader[self.currt_index]
            images_batch = images_batch.permute(0, 1, 4, 2, 3)
            self.images_batch = images_batch
            self.gt_batch = gt_batch
            self.intrinsics_batch = intrinsics_batch
            self.poses_batch = poses_batch
            self.frame_id = frame_id
            self.currt_index = self.currt_index + 1
        else:
            self.images_batch = None
            self.gt_batch = None
            self.intrinsics_batch = None
            self.poses_batch = None
            self.frame_id = None
            self.reset()
            return 
        with torch.cuda.stream(self.stream):
            self.images_batch = self.images_batch.cuda(non_blocking=True).float()
            self.gt_batch = self.gt_batch.cuda(non_blocking=True)
            self.intrinsics_batch = self.intrinsics_batch.cuda(non_blocking=True).float()
            self.poses_batch = self.poses_batch.cuda(non_blocking=True)


    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        images_batch = self.images_batch
        gt_batch = self.gt_batch
        poses_batch = self.poses_batch
        intrinsics_batch = self.intrinsics_batch
        frame_id = self.frame_id
        self.preload()
        return images_batch, poses_batch, gt_batch, intrinsics_batch, frame_id

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
        is_use_gpu = True
        torch.autograd.set_detect_anomaly(True)
        # 获取深度步长
        batch_size = num_gpus * cfg.TRAIN.BATCH[stage-1]
        # 获取最大步长
        max_steps = cfg.TRAIN.ITERS[stage-1]
        # 设置GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.TRAIN.USE_GPU
        # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 设置存储频率
        SUMMARY_FREQ = 10
        # 设置日志频率
        LOG_FREQ = 50
        # 设置checkpoint中间输出频率
        CHECKPOINT_FREQ = 100
        # 设置最大步长
        self.training_steps = max_steps
        # 开始加载数据模型
        print ("batch size: %d \t max steps: %d"%(batch_size, max_steps))
        # 初始化网络
        deepModel = DepthModule(cfg)
        # 定义损失函数
        loss_function = MyLoss(deepModel)
        
        if is_use_gpu:
            if torch.cuda.device_count() > 1:
                deepModel = nn.DataParallel(deepModel)
            deepModel = deepModel.to(device)
            loss_function.to(device)

        # 计算loss值
        running_loss = 0.0
        start_step = 0
        end_step = max_steps
        # # # 设置损失函数
        # optimizer = optim.Adam(deepModel.parameters(), lr=cfg.TRAIN.LR)
        # # 设置学习策略
        # model_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, int(max_steps), 0.9)
       
        # # 设置损失函数
        optimizer = optim.RMSprop(deepModel.parameters(), lr=cfg.TRAIN.LR, momentum=0.9)
        # #  # 设置学习策略
        model_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 5000, 0.6)
        
        # 设置训练数据集
        trainloader = torch.utils.data.DataLoader(data_source, batch_size=batch_size, shuffle=True,
                            num_workers=32, pin_memory=True, drop_last=True)
        # 设置为训练模式
        deepModel.train()
        # 加载模型
        if cfg.STORE.IS_USE_RESRORE:
            # 加载模型
            checkpoint = torch.load(cfg.STORE.RESRORE_PATH)
            deepModel.load_state_dict(checkpoint['net'])
            #optimizer.load_state_dict(checkpoint['optimizer'])
            start_step = checkpoint['epoch']
            #model_lr_scheduler.load_state_dict(checkpoint['model_lr_scheduler'])
            end_step = start_step + max_steps
        # for p in optimizer.param_groups:
        #     p['lr'] = cfg.TRAIN.LR
        
        
        # 日志
        if cfg.STORE.IS_SAVE_LOSS_LOG:
            loss_log_file_name = os.path.join(cfg.LOG_DIR, time.strftime("%Y%m%d%H%M%S.log", time.localtime()))
            if not os.path.exists(loss_log_file_name):
                os.mknod(loss_log_file_name)
            loss_file = open(loss_log_file_name, "w") 
            writer = SummaryWriter(cfg.LOG_DIR+"_sum")
        else:
            loss_file = None
            writer = None
        prefetcher = data_prefetcher(cfg, trainloader)
        data_len = len(trainloader)
        #prefetcher =  data_prefetcher(trainloader)
        for training_step in range(start_step, end_step):
            delta = 0
            abs_rel = 0
            #prefetcher = data_prefetcher(cfg, trainloader)
            images_batch, poses_batch, gt_batch, intrinsics_batch, frame_id = prefetcher.next()
            #print(len(trainloader))
            i = 0
            # 开始加载数据
            while i < data_len:
            #for i, data in enumerate(trainloader, 0):
            #while images_batch is not None:
                # 进行数据预处理

                #images_batch, poses_batch, gt_batch, myfilled, myfilled, intrinsics_batch, frameid = data
                #images_batch, gt_batch, intrinsics_batch, a = prepare_inputs(cfg , images_batch, gt_batch, intrinsics_batch)
                optimizer.zero_grad()
                # images_batch = images_batch.permute(0, 1, 4, 2, 3)
                # Ts = poses_batch.cuda()
                # images = images_batch.cuda()
                # intrinsics_batch = intrinsics_batch.cuda().float()
                # gt_batch = gt_batch.cuda()

                Ts = poses_batch
                images = images_batch
                intrinsics = intrinsics_batch
                gt = gt_batch
                outputs = deepModel(
                    Ts, 
                    images, 
                    intrinsics
                    )
                # 计算loss值
                loss = loss_function(gt, outputs)
                # 计算误差
                # loss backward
                loss.backward()
                # update parameters using optimizer
                optimizer.step()
                #torch.cuda.empty_cache()
                # 计算loss
                #running_loss = running_loss + float(loss)
                temp_delta, temp_abs_rel = add_depth_acc(gt.detach(), outputs.detach())
                abs_rel = abs_rel + float(temp_abs_rel)
                delta = delta + float(temp_delta)
                i = i + 1
                images_batch, poses_batch, gt_batch, intrinsics_batch, frame_id = prefetcher.next()
                # 进行文件写入
            if (writer is not None) and (training_step % SUMMARY_FREQ == 0) :
                writer.add_scalar('Train/Loss', loss.detach().cpu().numpy(), training_step)
                writer.add_scalar('Train/abs_rel', abs_rel/data_len, training_step)
                writer.add_scalar('Train/a1', delta/data_len, training_step)
                writer.add_scalar('Train/lr', float(model_lr_scheduler.get_last_lr()[0]), training_step)


            # 修改学习率
            model_lr_scheduler.step()
            # 输出loss值
            if training_step % LOG_FREQ == 0:
                loss_str = "[step= {:>5d}] loss: {:.9f}".format(training_step, loss.detach().cpu().numpy() )
                print(loss_str)
                # 需要记录loss值
                if loss_file is not None:
                    loss_file.writelines(loss_str+"\n") 
                    loss_file.flush()
                running_loss = 0.0
            # 输出模型,注意这里主要是checkpoint的保存
            if training_step % CHECKPOINT_FREQ == 0:
                # 模型名称
                end = "step_{}.pth".format(training_step)
                save_file = os.path.join(cfg.CHECKPOINT_DIR, end)
                
                checkpoint = {
                    "net": deepModel.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": training_step,
                    "model_lr_scheduler": model_lr_scheduler.state_dict()
                }
                # 模型名字
                torch.save(checkpoint, save_file)
        # 最后进行一次模型保存
        end = "final.pth"
        save_file = os.path.join(cfg.CHECKPOINT_DIR, end)
        
        checkpoint= {
            "net": deepModel.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": end_step,
            "model_lr_scheduler": model_lr_scheduler
            }
        # 模型名字
        torch.save(checkpoint, save_file)
        if loss_file is not None:
            loss_file.close()




