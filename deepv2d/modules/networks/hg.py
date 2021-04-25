import torch
import torch.nn as nn

import numpy as np
from .layer_ops import *  


class Hourglass2d(nn.Module):
    def __init__(self, nIn, nOut, stack_number, expand=64):
        super().__init__()
        self.dim2 = nOut + expand
        # 第一次卷积
        self.resnet = ResConv2d(nIn, nOut)
        # 最大池化层,注意输入和输出
        self.pool = MaxPool2D(nOut, nOut, 2)
        # 进行向上3*3卷积 
        self.low1_conv = Conv2dBnRel(nOut, self.dim2, 3)

        if stack_number > 1:
            self.low2_conv = Hourglass2d(self.dim2, self.dim2, stack_number-1)
        else:
            self.low2_conv = Conv2dBnRel(self.dim2, self.dim2, 3)
        # 重新进行卷积
        self.low3_conv = Conv2dBnRel(self.dim2, nOut, 3)
        # 进行上采样
        #self.upnn2d = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, input):
        """
        进行前向计算
        Args:
            input ([type]): [description]
        """
        out = input + self.resnet(input)
        ht, wd = out.shape[-2:]
        pool1 = self.pool(out)
        low1 = self.low1_conv(pool1)
        low2 = self.low2_conv(low1)
        low3 = self.low3_conv(low2)
        # up data
        up2 = nn.functional.upsample_nearest(low3, size=(ht, wd))
        out += up2
        return out

class FastHourglass2d(nn.Module):
    def __init__(self, nIn, nOut, stack_number, expand=64):
        super().__init__()
        self.dim2 = nOut + expand
        # 第一次卷积,1*1 快速卷积
        self.resnet = Conv2dBnRel(nIn, nOut, 1)
        # 最大池化层,注意输入和输出
        self.pool = MaxPool2D(nOut, nOut, 2)
        # 进行向上3*3卷积 
        self.low1_conv = Conv2dBnRel(nOut, self.dim2, 3)

        if stack_number > 1:
            self.low2_conv = FastHourglass2d(self.dim2, self.dim2, stack_number-1)
        else:
            self.low2_conv = Conv2dBnRel(self.dim2, self.dim2, 3)
        # 重新进行卷积
        self.low3_conv = Conv2dBnRel(self.dim2, nOut, 1)
        # 进行上采样
        #self.upnn2d = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, input):
        """
        进行前向计算
        Args:
            input ([type]): [description]
        """
        out = input + self.resnet(input)
        ht, wd = out.shape[-2:]
        pool1 = self.pool(out)
        low1 = self.low1_conv(pool1)
        low2 = self.low2_conv(low1)
        low3 = self.low3_conv(low2)
        # up data
        up2 = nn.functional.upsample_nearest(low3,size=(ht, wd))
        out += up2
        return out

# hg = Hourglass2d(32,32,2)

# input = torch.randn(5, 32, 240, 320)
# a = hg(input)
# print(a)

class Hourglass3d(nn.Module):
    def __init__(self, nIn, nOut, stack_number, expand=64):
        super().__init__()
        dim2 = nOut + expand
        # 第一次卷积
        self.resnet = ResConv3d(nIn, nOut)
        # 最大池化层,注意输入和输出
        self.pool = MaxPool3D(nOut, nOut, 2)
        # 进行向上3*3卷积 
        self.low1_conv = Conv3dBnRel(nOut, dim2, 3)

        if stack_number > 1:
            self.low2_conv = Hourglass3d(dim2, dim2, stack_number-1)
        else:
            self.low2_conv = Conv3dBnRel(dim2, dim2, 3)
        # 重新进行卷积
        self.low3_conv = Conv3dBnRel(dim2, nOut, 3)
       
    def forward(self, input):
        """
        进行前向计算
        Args:
            input ([type]): [description]
        """
        out = self.resnet(input) # 1 32 32 30 40 
        dt, ht, wd = out.shape[-3:]
        pool1 = self.pool(out) # 1 32 16 15 20 
        low1 = self.low1_conv(pool1) # 1 96 16 45 20 
        low2 = self.low2_conv(low1)
        low3 = self.low3_conv(low2)
        # up data
        up2 = nn.functional.upsample_nearest(low3, size=(dt, ht, wd))
        out = out + up2
        return out


class FastHourglass3d(nn.Module):
    def __init__(self, nIn, nOut, stack_number, expand=64):
        super().__init__()
        self.dim2 = nOut + expand
        # 第一次卷积
        self.resnet = Conv3d(nIn, nOut,1)
        # 最大池化层,注意输入和输出
        self.pool = MaxPool3D(nOut, nOut, 2)
        # 进行向上3*3卷积 
        self.low1_conv = Conv3dBnRel(nOut, self.dim2, 3)

        if stack_number > 1:
            self.low2_conv = Hourglass3d(self.dim2, self.dim2, stack_number-1)
        else:
            self.low2_conv = Conv3dBnRel(self.dim2, self.dim2, 3)
        # 重新进行卷积 注意这里是1*1
        self.low3_conv = Conv3dBnRel(self.dim2, nOut, 1)
        # 进行上采样
        self.upnn2d = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, input):
        """
        进行前向计算
        Args:
            input ([type]): [description]
        """
        out = input + self.resnet(input)
        dt, ht, wd = out.shape[-3:]
        pool1 = self.pool(out)
        low1 = self.low1_conv(pool1)
        low2 = self.low2_conv(low1)
        low3 = self.low3_conv(low2)
        # up data
        up2 = nn.functional.upsample_nearest(low3, size=(dt, ht, wd))
        out = out + up2
        return out


# hg = FastHourglass3d(32,32,2)

# input = torch.randn(1, 32, 32, 240, 320)
# a = hg(input)
# print(a)


class Aspp2d(nn.Module):
    def __init__(self, nIn, nOut, expand=64):
        """
        aspp尺度卷积网络，主要是为了抵抗尺度变换

        Args:
            nIn ([type]): 输入维度
            dim ([type]): [description]
            expand (int, optional): [description]. Defaults to 64.

        Returns:
            [type]: [description]
        """
        super().__init__()
        start_d = 6
        # 计算输出维度
        out_dim = nOut + expand
        # 一维卷积
        self.conv1 = Conv2d(nIn, out_dim, 1)
        # 空洞卷积1
        self.dilate_conv1 = CDilated(nIn, out_dim, 3, d = start_d*1)
        # 空洞卷积2
        self.dilate_conv2 = CDilated(nIn, out_dim, 3, d = start_d*2)
        # 空洞卷积3
        self.dilate_conv3 = CDilated(nIn, out_dim, 3, d = start_d*3)
        # 最后的卷积操作,注意这里的输出操作
        self.last_conv = Conv2dBnRel(out_dim*4, nOut, 1)
        

    def forward(self, input):
        """
        进行前向计算
        Args:
            input ([type]): 输入数据
        """
        data1 = self.conv1(input)
        data2 = self.dilate_conv1(input)
        data3 = self.dilate_conv2(input)
        data4 = self.dilate_conv3(input)
        out = self.last_conv(torch.cat([data1, data2, data3,data4], dim=1))
        return out

# hg = Aspp2d(32,32)

# input = torch.randn(5, 32, 240, 320)
# a = hg(input)
# print(a)
