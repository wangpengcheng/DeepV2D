import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import cv2
import torch.optim as optim
<<<<<<< HEAD

from geometry.transformation import *

from modules.depth_module import DepthModule


=======
from torchsummary import summary
from geometry.transformation import *

from modules.depth_module import DepthModule
from modules.my_loss import MyLoss


class data_prefetcher():
    def __init__(self, loader):
        # 初始化cuda
        self.loader = iter(loader)
        # 运算流
        self.stream = torch.cuda.Stream()
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        # 预加载图像
        self.preload()

    def preload(self):
        try:
            #self.next_input, self.next_target = next(self.loader)
            self.images_batch, poses_batch, self.gt_batch, filled_batch, pred_batch, self.intrinsics_batch, frame_id = next(self.loader)
        except StopIteration:
            self.images_batch = None
            self.gt_batch = None
            self.intrinsics_batch = None
            return
        with torch.cuda.stream(self.stream):
            self.images_batch = self.images_batch.cuda(non_blocking=True)
            self.gt_batch = self.gt_batch.cuda(non_blocking=True)
            self.intrinsics_batch = self.intrinsics_batch.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.images_batch = self.images_batch.permute(0, 1, 4, 2, 3)
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)  
        # features = self.next_features_gpu
        # targets = self.next_targets_gpu
        features = self.next_features
        targets = self.next_targets
        if features is not None:
            features = [xaf.record_stream(torch.cuda.current_stream()) for xaf in features]
        if targets is not None:
            targets = [targets[xaf].record_stream(torch.cuda.current_stream()) for xaf in targets.keys()]
        self.preload()
        return features, targets
>>>>>>> 5381cefe6ddf5719e983acecf30288546f1e7034

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
<<<<<<< HEAD

=======
>>>>>>> 5381cefe6ddf5719e983acecf30288546f1e7034
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
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 设置存储频率
        SUMMARY_FREQ = 10
        # 设置日志频率
        LOG_FREQ = 100
        # 设置checkpoint中间输出频率
        CHECKPOINT_FREQ = 500
        # 设置最大步长
        self.training_steps = max_steps
        # 开始加载数据模型
        print ("batch size: %d \t max steps: %d"%(batch_size, max_steps))
        # 初始化网络
        deepModel = DepthModule(cfg)
        # 定义损失函数
        loss_function = MyLoss(deepModel)
         # 加载模型
        if restore_ckpt is not None:
            # 加载模型
            deepModel.load_state_dict(torch.load(restore_ckpt))
        if is_use_gpu:
            if torch.cuda.device_count() > 1:
                deepModel = nn.DataParallel(deepModel)
            deepModel = deepModel.cuda()
            loss_function.cuda()


        
        # 设置损失函数
        optimizer = optim.SGD(deepModel.parameters(), lr=cfg.TRAIN.LR, momentum=0.9)
        # 设置学习策略
        model_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, max_steps, 0.1)
        # 计算loss值
        running_loss = 0.0

        # 设置训练数据集
        trainloader = torch.utils.data.DataLoader(data_source, batch_size=batch_size, shuffle=False, num_workers=8)
        # 设置为训练模式
        deepModel.train()
        #prefetcher =  data_prefetcher(trainloader)
        for training_step in range(max_steps):
            
            # 开始加载数据
            for i, data in enumerate(trainloader, 0):
                
                images_batch, poses_batch, gt_batch, filled_batch, pred_batch, intrinsics_batch, frame_id= data
                #images_batch, gt_batch, intrinsics_batch =  prefetcher.next()
                # 进行数据预处理,主要是维度交换
                images = images_batch.permute(0, 1, 4, 2, 3)
                
                #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                #deepModel = deepModel.to(device)
                # set optimizer buffer to 0
                optimizer.zero_grad()
                #summary(deepModel, [(Ts.shape[0:]), (images.shape[0:]), (intrinsics_batch.shape[0:])])
                # 前向计算
              
                Ts = poses_batch.cuda()
                images = images.cuda()
                intrinsics_batch = intrinsics_batch.cuda()
                gt_batch = gt_batch.cuda()

                outputs = deepModel(
                    Ts, 
                    images, 
                    intrinsics_batch
                    )
                

                # 计算loss值
                loss = loss_function(gt_batch, outputs)
                # loss backward
                loss.backward()
                # update parameters using optimizer
                optimizer.step()
                # 计算loss
                running_loss = running_loss + float(loss.detach().item())

            # 修改学习率
            model_lr_scheduler.step()
            # 输出loss值
            if training_step % LOG_FREQ == 0:
                print('[step=%5d] loss: %.9f'%(training_step, running_loss))
                running_loss = 0.0
            # 输出模型
            if training_step % CHECKPOINT_FREQ == 0 or training_step+1 == max_steps:
                # 模型名称
                save_file = os.path.join(cfg.CHECKPOINT_DIR, cfg.TRAIN.MODULE_NAME)
                # 模型名字
                torch.save(deepModel.state_dict(), save_file)




