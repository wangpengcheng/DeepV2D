
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

#自定义损失函数

# 1. 继承nn.Mdule
class MyLoss(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.transform = transforms.Compose([transforms.Resize(size=(net.ht, net.wd))])
        
    def forward1(self, depth_gt, outputs,log_error=True):
        # 获取形状
        total_loss = 0.0
        # 遍历推理出来的深度图想
        for i, logits in enumerate(self.net.pred_logits):
            # 进行归一化
            pred = self.net.soft_argmax(logits)
            # 将归一化的图像线性插值
            pred = self.transform(pred)
            # 所有深度大于0的值
            zero = torch.zeros_like(depth_gt)
            valid = torch.where(depth_gt > 0.0, zero, depth_gt).float()
            # 计算所有元素的均值
            s = 1.0 / (torch.mean(valid) + 1e-8)

            gx = pred[:, :, 1:] - pred[:, :, :-1]
            gy = pred[:, 1:, :] - pred[:, :-1, :]
            vx = valid[:, :, 1:] * valid[:, :, :-1]
            vy = valid[:, 1:, :] * valid[:, :-1, :]

            # take l1 smoothness loss where gt depth is missing
            loss_smooth = \
                torch.mean((1-vx)*torch.abs(gx)) + \
                torch.mean((1-vy)*torch.abs(gy))

            loss_depth = s*torch.mean(valid*torch.abs(depth_gt-pred))
            loss_i = self.net.cfg.STRUCTURE.TRAIN.SMOOTH_W * loss_smooth + loss_depth

            w = .5**(len(self.net.pred_logits)-i-1)
            total_loss += w * loss_i

        # if log_error:
        #     #add_depth_summaries(gt, pred)
        
        return total_loss
        
    def forward(self, depth_gt, outputs, log_error=True):
        # 获取形状
        total_loss = 0.0
        # 将归一化的图像线性插值
        pred = self.transform(outputs)
        # 所有深度大于0的值
        zero = torch.zeros_like(depth_gt)
        valid = torch.where(depth_gt > 0.0, depth_gt, zero).float()
        # 计算所有元素的均值
        s = 1.0 / (torch.mean(valid) + 1e-8)

        gx = pred[:, :, 1:] - pred[:, :, :-1]
        gy = pred[:, 1:, :] - pred[:, :-1, :]
        vx = valid[:, :, 1:] * valid[:, :, :-1]
        vy = valid[:, 1:, :] * valid[:, :-1, :]

        # take l1 smoothness loss where gt depth is missing
        loss_smooth = \
            torch.mean((1-vx)*torch.abs(gx)) + \
            torch.mean((1-vy)*torch.abs(gy))

        loss_depth = s*torch.mean(valid*torch.abs(depth_gt-pred))
        loss_i = self.net.cfg.STRUCTURE.TRAIN.SMOOTH_W * loss_smooth + loss_depth

        total_loss += loss_i

        # if log_error:
        #     #add_depth_summaries(gt, pred)
        
        return total_loss
    
class LightLoss(nn.Module):
    def __init__(self, height = 240, width = 320, smooth_w = 0.02 ):
        super().__init__()
        self.smooth = smooth_w 
        self.ht = height 
        self.wd = width
        self.transform = transforms.Compose([transforms.Resize(size=(self.ht, self.wd))])
        
        
    def forward(self, depth_gt, outputs, log_error=True):
        # 获取形状
        total_loss = 0.0
        # 将归一化的图像线性插值
        pred = self.transform(outputs)
        # 所有深度大于0的值
        zero = torch.zeros_like(depth_gt)
        valid = torch.where(depth_gt > 0.0, depth_gt, zero).float()
        # 计算所有元素的均值
        s = 1.0 / (torch.mean(valid) + 1e-8)

        gx = pred[:, :, 1:] - pred[:, :, :-1]
        gy = pred[:, 1:, :] - pred[:, :-1, :]
        vx = valid[:, :, 1:] * valid[:, :, :-1]
        vy = valid[:, 1:, :] * valid[:, :-1, :]

        # take l1 smoothness loss where gt depth is missing
        loss_smooth = \
            torch.mean((1-vx)*torch.abs(gx)) + \
            torch.mean((1-vy)*torch.abs(gy))

        loss_depth = s*torch.mean(valid*torch.abs(depth_gt-pred))
        loss_i = 0.02 * loss_smooth + loss_depth

        total_loss += loss_i

        # if log_error:
        #     #add_depth_summaries(gt, pred)
        
        return total_loss