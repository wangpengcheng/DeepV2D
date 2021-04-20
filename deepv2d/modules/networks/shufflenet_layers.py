import torch 
import numpy as np
import torch.nn as nn
from .layer_ops import *  
# https://blog.csdn.net/qq_38675397/article/details/104249654


def split(x, groups):
    out = x.chunk(groups, dim=1) 
    return out

def channel_shuffle(inputs, num_groups):
    """
    通道混洗
    Args:
        inputs ([type]): 输入数据
        num_groups ([type]): 分组数量

    Returns:
        [type]: 返回混洗数据
    """
    # 获取形状
    N, C, H, W = inputs.size()
    # 进行分组和通道交换等操作
    out = inputs.view(N, num_groups, C // num_groups, H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)
 
    return out

class GroupsConvBN(nn.Module):

    def __init__(self, inputs, filters, kernel, num_groups, strides=1):

        """
        进行组卷积
        Args:
            inputs ([type]): 输入维度
            filters ([type]): 输出维度
            kernel ([type]): 卷积核大小
            strides ([type]): 步长
            num_groups ([type]): 分组数量

        Returns:
            Tensor: 卷积操作结果
        """
        super().__init__()
        # 计算padd
        padding = int((kSize - 1)/2)
        self.group_conv = nn.Conv2d(inputs, filters, kernel_size=kernel, groups=num_groups, bias=False)
        self.bn = nn.BatchNorm2d(filters)
        self.act = nn.ReLU()

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.group_conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output

class DepthSeparableConvBN(nn.Module):
    def __init__(self, in_dims, out_dims, stride = 1):
        """
        深度可分离卷积类
        Args:
            in_dims ([type]): [description]
            out_dims ([type]): [description]
            stride (int, optional): [description]. Defaults to 1.
        """
        super(DepthSeparableConvBN, self).__init__()
        # groups = nInputPlane,就是Depthwise,如果groups = nInputPlane,kernel=(K, 1)（针对二维卷积,前面加上,groups=1 and kernel=(1, K)）,就是可分离的。
        self.depth_wise = nn.Conv2d(in_dims, in_dims, kernel_size=3, stride = stride, padding = 1, groups = in_dims, bias=False)
        self.point_wise = nn.Conv2d(in_dims, out_dims, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_dims)

    def forward(self, x):
        out = self.depth_wise(x)
        out = self.point_wise(out)
        out = self.bn(out)
        return out


class ShuffleNetUnitA(nn.Module):
    def __init__(self, input_dims, out_channels, num_groups):
        """
        shuffle net 卷积单元1
        Args:
            inputs ([type]): 输入数据
            out_dims : 输出维度数据
            num_groups ([type]): 分组数量

        Returns:
            [type]: 返回的值
        """
        super().__init__()
        
        # 块通道数目,即分组数量,一般都是4
        bottleneck_channels = int(out_channels // 4)
        # 设置卷积通道数
        self.num_groups = num_groups
        # 分组卷积
        self.group_conv1 = GroupsConvBN(input_dims, bottleneck_channels, kernel=1, num_groups=num_groups)
        # 注意这里的通道混洗
        
        # 深度可分离卷积
        self.depthwise_conv_bn = DepthSeparableConvBN(bottleneck_channels, bottleneck_channels, stride = 1)
        # 执行分组卷积
        self.group_conv2 = GroupsConvBN(bottleneck_channels, out_channels, kernel=1, num_groups=num_groups)
        # 最后激活层
        self.active = nn.ReLU()
    def forward(self, input):
        # 组卷积
        out = self.group_conv1(input)
        # 通道混洗
        out = channel_shuffle(out, self.num_groups)
        # 进行通道混洗
        out = self.depthwise_conv_bn(out)
        # 进行分组卷积
        out = self.group_conv2(out)
        # 进行添加
        out += input
        # 激活函数
        out = self.active(out)
        return out

class ShuffleNetUnitB(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups):
        """
        shuffle net 卷积单元2
        Args:
            inputs ([type]): 输入通道数目
            out_dims : 输出维度数据
            num_groups ([type]): 分组数量

        Returns:
            [type]: 返回的值
        """
        super().__init__()
        # # 将输出通道数目减去输入通道数目？？？,需要其它的来进行池化
        out_channels -= in_channels
        # 块通道数目,即分组数量,一般都是4
        bottleneck_channels = int(out_channels // 4)
        # 设置卷积通道数
        self.num_groups = num_groups
        # 分组卷积
        self.group_conv1 = GroupsConvBN(in_channels, bottleneck_channels, kernel=1, num_groups=num_groups)
        # 通道混洗
        # 深度可分离卷积
        self.depthwise_conv_bn = DepthSeparableConvBN(bottleneck_channels, bottleneck_channels, stride = 2)
        # 执行分组卷积
        self.group_conv2 = GroupsConvBN(bottleneck_channels, out_channels, kernel=1, num_groups=num_groups)
        # 最后激活层
        self.active = nn.ReLU()

    def forward(self, input):
        residual = input
        out = self.group_conv1(input)
        # 进行通道混洗
        out = channel_shuffle(out, self.num_groups)
        # 进行通道混洗
        out = self.depthwise_conv_bn(out)
        # 进行分组卷积
        out = self.group_conv2(out)
        # 进行均值池化
        residual = nn.functional.avg_pool2d(residual, kernel_size=3,stride=2,padding=1)
        # 进行添加
        out = torch.cat((residual, out), dim=1) # [N,C,H,W],C cat
        # 进行激活函数
        out = self.active(out)
        return out


class ShuffleNetUnitV2A(nn.Module):
    def __init__(self, input_dims, out_channels, num_groups):
        """
        shuffle net 卷积单元1
        Args:
            inputs ([type]): 输入数据
            out_dims : 输出维度数据
            num_groups ([type]): 分组数量

        Returns:
            [type]: 返回的值
        """
        super().__init__()
        assert out_channels % 2 == 0
        # 块通道数目,即分组数量,一般都是4
        mid_channels = int(out_channels // 2)
        # 分组卷积
        self.conv1 = Conv2dBnRel(input_dims, mid_channels, kSize=1, stride=1)
        # 注意这里的通道混洗
        
        # 深度可分离卷积
        self.depthwise_conv_bn = DepthSeparableConvBN(mid_channels, mid_channels, stride = 1)
        # 执行分组卷积
        self.conv2 = Conv2dBnRel(mid_channels, mid_channels, kSize=1, stride=1)
        # 最后激活层
        self.active = nn.ReLU()

    def forward(self, input):
        # 首先进行二维分离操作,将第三维分成两个部分
        shortcut, x = input.chunk(2, dim=1)
        # 组卷积
        out = self.conv1(input)
        # 进行深度可分离卷积
        out = self.depthwise_conv_bn(out)
        # 进行分组卷积
        out = self.conv2(out)
        # 进行数据合并
        out = torch.cat((out, shortcut), dim = 1)
        # 进行通道混洗
        out = channel_shuffle(out, 2)
        return out


class ShuffleNetUnitV2B(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups):
        """
        shuffle net 卷积单元2
        Args:
            inputs ([type]): 输入通道数目
            out_dims : 输出维度数据
            num_groups ([type]): 分组数量

        Returns:
            [type]: 返回的值
        """
        super().__init__()
        assert out_channels % 2 == 0
        # 块通道数目,即分组数量,一般都是4
        mid_channels = int(out_channels // 2)
        # 设置卷积通道数
        self.num_groups = num_groups
        # 分组卷积
        self.conv1 = Conv2dBnRel(in_channels, mid_channels, kSize=1, stride=1)
        # 深度可分离卷积
        self.depthwise_conv_bn = DepthSeparableConvBN(mid_channels, mid_channels, stride = 2)
        # 执行分组卷积
        self.conv2 = Conv2dBnRel(mid_channels, out_channels - in_channels, kSize=1, stride=1)
        # 在进行分组卷积
        self.depthwise_conv_bn_01 =  DepthSeparableConvBN(in_channels, in_channels, stride = 2)
        # 再进行卷积
        self.conv3 = Conv2dBnRel(mid_channels, in_channels, kSize=1, stride=1)
        # 最后激活层
        self.active = nn.ReLU()

    def forward(self, input):
        shortcut = input
        # 进行一次卷积
        out = self.conv1(input)
        # 
        out = self.depthwise_conv_bn(out)
        # 
        out = self.conv2(out)
        # 另外一个分支进行操作
        shortcut = self.depthwise_conv_bn_01(shortcut)
        shortcut = self.conv3(shortcut)
        out = torch.cat((out, shortcut), dim = 1)
        # 进行通道混洗
        out = channel_shuffle(out, 2)

        return out



        

# a = ShuffleNetUnitV2B(32,64,2)
# b = ShuffleNetUnitV2A(64,64,2)


# input = torch.randn(5, 32, 240, 320)
# a_out = a(input)
# b_out = b(a_out)
# print(a_out)