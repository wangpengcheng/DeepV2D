import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from .layer_ops import * 
from .hg import * 
from .shufflenet_layers import *

# tf 1*4*480*640*3 NHWC batch, frames, ht, wd, channel
# pytorch b, N，C, D,H,W batch,n, channel, d, ht, wd 相当于-1 和1的维度交换了
class StereoHead(nn.Module):
    def __init__(self, inputs_dims, out_size):
        super().__init__()
        self.conv1 = Conv3d(inputs_dims, 32, 3)
        # 激活函数
        self.act = nn.ReLU()
        self.conv2 = Conv3d(32, 32, 3)

        self.conv3 = Conv3d(32, 1, 1)
        # 注意这里默认的是线性插值
        self.transform = transforms.Compose([transforms.Resize(size=out_size)])
    
    def forward(self, input):
        out = self.conv1(input) 
        out = self.act(out)
        out = self.conv2(out)
        out = self.act(out)
        out = self.conv3(out)
        # 降低维度操作,注意这里是真毒 
        out = torch.squeeze(out, dim = 1)
        out = self.transform(out)
        return out

class FastStereoHead(nn.Module):
    def __init__(self, inputs_dims, out_size):
        super().__init__()
        # 1*1
        self.conv1_1 = Conv3d(inputs_dims, 32, 1)
        # 激活函数
        self.act = nn.ReLU()
        # 3*3
        self.conv2 = Conv3d(32, 32, 3)

        self.conv3 = Conv3d(32, 32, 1)
        self.conv4 = Conv3d(32, 1, 1)
        # 注意这里默认的是线性插值
        self.transform = transforms.Compose([transforms.Resize(size=out_size)])
    
    def forward(self, input):
        out = self.conv1_1(input) 
        out = self.act(out)
        out = self.conv2(out)
        out = self.act(out)
        out = self.conv3(out)
        out = self.act(out)
        out = self.conv4(out)

        # 降低维度操作,注意这里是真毒 
        out = torch.squeeze(out, dim = 1)
        out = self.transform(out)
        return out



class ResnetEncoder(nn.Module):
    def __init__(self, inputs_dims, output_dims, stack_count=1 ,depth_count=2):
        super().__init__()
        """
        基础编码单元
        Args:
            inputs_dims ([type]): 输入维度
            output_dims ([type]): 输出维度
            stack_count: 沙漏网络的个数
            depth_count: 沙漏网络的深度
        """
        self.depth_count = depth_count
        self.conv1 = Conv2d(inputs_dims, 32, 7, stride = 2)
        self.res_conv1 = ResConv2d(32, 32, 1)
        self.res_conv2 = ResConv2d(32, 32, 1)
        self.res_conv3 = ResConv2d(32, 32, 1)
        self.res_conv4 = ResConv2d(32, 64, 2)
        self.res_conv5 = ResConv2d(64, 64, 1)
        self.res_conv6 = ResConv2d(64, 64, 1)
        self.res_conv7 = ResConv2d(64, 64, 1)

        self.stack_conv = Hourglass2d(64, 64, depth_count)
        self.conv2 = Conv2d(64, output_dims, 1, stride = 1)

    def forward(self, input):
        """
        进行前向计算
        Args:
            input ([type]): [description]
        """
        # 进行卷积
        out = self.conv1(input)
        out = self.res_conv1(out)
        out = self.res_conv2(out)
        out = self.res_conv3(out)
        out = self.res_conv4(out)
        out = self.res_conv5(out)
        out = self.res_conv6(out)
        out = self.res_conv7(out)
        # 沙漏网络
        for i in range(stack_count):
            out = Hourglass2d(out)

        # 最后卷积
        out = self.conv2(out)
        return out

class ResnetDecoder(nn.Module):
    def __init__(self, inputs_dims, output_dims, pred_logits, out_size, stack_count= 1 ,depth_count= 2):
        """
        基础解码单元
        Args:
            inputs_dims ([type]): 输入维度
            output_dims ([type]): 输出维度
            stack_count: 沙漏网络的个数
            depth_count: 沙漏网络的深度
            pred_logits: 预测队列数组
            out_size: 图像大小
        """
        super().__init__()
        self.pred_logits = pred_logits
        self.stack_count = stack_count
        self.depth_count = depth_count
        self.conv1 = Conv3d(inputs_dims, 32, 1)
        self.res_conv1 = ResConv3d(32, 32)
        
        # 沙漏网络
        self.stack_conv = Hourglass3d(32, 32, depth_count)
        # 进行头部合并
        self.stereo_head = StereoHead(32, out_size)

    def forward(self, input):
        """
        进行前向计算
        Args:
            input ([type]): [description]
        """
        dims = input.shape
        # 重新进行数据
        volume = torch.reshape(input, [dims[0]*dims[1], 64, dims[3], dims[4], dims[5]])
        # 进行卷积
        out = self.conv1(volume)
        out = self.res_conv1(out) # 4 32 16 15 20
        # 重新整理输出维度为32 维度, 1*4*32*32*30*10
        out = torch.reshape(out, [dims[0],  dims[1], 32, dims[3], dims[4], dims[5]])
        # 求解均值
        out = torch.mean(out, dim = 1)
        # 沙漏网络
        for i in range(self.stack_count):
            out = self.stack_conv(out)
            out = self.stereo_head(out) # 1*32*240*320
            #self.pred_logits.append(out)
        return out 

class Shufflenetv2Encoder(nn.Module):
    def __init__(self, inputs_dims, output_dims, stack_count=1 ,depth_count=2):
        """
        基础编码单元
        Args:
            inputs_dims ([type]): 输入维度
            output_dims ([type]): 输出维度
            stack_count: 沙漏网络的个数
            depth_count: 沙漏网络的深度
        """
        super().__init__()
        # 第一次缩放卷积
        self.conv1 = Conv2d(inputs_dims, 32, 3, stride = 2)
        self.res_conv1 = ShuffleNetUnitV2A(32, 32, 2)
        self.res_conv2 = ShuffleNetUnitV2A(32, 32, 2)
        self.res_conv3 = ShuffleNetUnitV2B(32, 64, 2)
        self.res_conv4 = ShuffleNetUnitV2A(64, 64, 2)
        self.res_conv5 = ShuffleNetUnitV2A(64, 64, 2)
        self.res_conv6 = ShuffleNetUnitV2B(64, 128, 2)
        self.res_conv7 = ShuffleNetUnitV2A(128, 128, 2)
        self.res_conv8 = ShuffleNetUnitV2A(128, 128, 2)
        self.conv2 = Conv2dBnRel(128, 64, 3, 1)
        # aspp
        self.aspp  = Aspp2d(64, 64)

        self.conv3 = Conv2d(64, output_dims, 1, stride = 1)

    def forward(self, input):
        """
        进行前向计算
        Args:
            input ([type]): [description]
        """
        # 进行卷积
        out = self.conv1(input)
        out = self.res_conv1(out)
        out = self.res_conv2(out)
        out = self.res_conv3(out)
        out = self.res_conv4(out)
        out = self.res_conv5(out)
        out = self.res_conv6(out)
        out = self.res_conv7(out)
        out = self.res_conv8(out)
        out = self.conv2(out)
        out = self.aspp(out)
        # 最后卷积
        out = self.conv3(out)
        return out


class FastResnetEncoder(nn.Module):
    def __init__(self, inputs_dims, output_dims, stack_count=1 ,depth_count=2):
        """
        基础编码单元
        Args:
            inputs_dims ([type]): 输入维度
            output_dims ([type]): 输出维度
            stack_count: 沙漏网络的个数
            depth_count: 沙漏网络的深度
        """
        super().__init__()
        self.stack_count = stack_count
        self.depth_count = depth_count

        self.conv1 = Conv2d(inputs_dims, 32, 3, stride = 2)
        self.res_conv1 = FastResConv2d(32, 32, 1)
        self.res_conv2 = FastResConv2d(32, 32, 1)
        self.res_conv3 = FastResConv2d(32, 32, 2)
        self.res_conv4 = FastResConv2d(32, 32, 1)
        self.res_conv5 = FastResConv2d(32, 32, 1)
        self.res_conv6 = FastResConv2d(32, 64, 2)
        self.res_conv7 = FastResConv2d(64, 64, 1)
        self.res_conv8 = FastResConv2d(64, 64, 1)

        self.stack_conv = FastHourglass2d(64, 64, depth_count)
        self.conv2 = Conv2d(64, output_dims, 1, stride = 1)
    def forward(self, input):
        """
        进行前向计算
        Args:
            input ([type]): [description]
        """
        # 进行卷积
        out = self.conv1(input)
        out = self.res_conv1(out)
        out = self.res_conv2(out)
        out = self.res_conv3(out)
        out = self.res_conv4(out)
        out = self.res_conv5(out)
        out = self.res_conv6(out)
        out = self.res_conv7(out)
        out = self.res_conv8(out)
        # 沙漏网络
        for i in range(self.stack_count):
            out = self.stack_conv(out)
        # 最后卷积
        out = self.conv2(out)
        return out


class FastResnetDecoder(nn.Module):
    def __init__(self, inputs_dims,pred_logits, out_size, stack_count=1 ,depth_count=2):
        """
        基础解码单元
        Args:
            inputs_dims ([type]): 输入维度
            output_dims ([type]): 输出维度
            stack_count: 沙漏网络的个数
            depth_count: 沙漏网络的深度
            pred_logits: 预测队列数组
            out_size: 图像大小
        """
        super().__init__()
        self.stack_count = stack_count
        self.depth_count = depth_count
        self.pred_logits = pred_logits
        self.conv1 = Conv3d(64, 32, 1, stride = 1)
        # 注意这里可以减少一个1*1 卷积
        self.res_conv1 = FastResConv3d(32, 32)
        
        # 沙漏网络
        self.stack_conv = FastHourglass3d(32, 32, depth_count)
        self.stereo_head = FastStereoHead(32, out_size)

    def forward(self, input):
        """
        进行前向计算
        Args:
            input ([type]): [description]
        """
        dims = input.shape
        # 重新进行数据 4*64*32*60*80
        volume = torch.reshape(input, [dims[0]*dims[1], 64, dims[3], dims[4], dims[5]])
        # 进行卷积 4*32*32*30*40   4*32*16*30*40 
        out = self.conv1(volume)
        out = self.res_conv1(out) #4*32*32*30*40
        # 重新整理输出维度为32 维度, 1*4*32*32*30*40 1 4 64 32 30 40
        out = torch.reshape(out,[dims[0],  dims[1], 32, dims[3], dims[4],dims[5]])
        # 求解均值 
        out = torch.mean(out, dim = 1)
        # 沙漏网络
        for i in range(self.stack_count):
            out = self.stack_conv(out)
            out = self.stereo_head(out)
            self.pred_logits.append(out)
        return out









<<<<<<< HEAD






=======
>>>>>>> 5381cefe6ddf5719e983acecf30288546f1e7034
