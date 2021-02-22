import torch 
import numpy as np
import torch.nn as nn

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
    out = inputs.view(N, num_groups, C // num_groups, H, W)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(N, C, H, W)
 
    return out

class GroupsConvBN(nn.Module):
    def __init__(self, inputs, filters, kernel, strides=1, num_groups):
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
        self.act = nn.ReLU(filters)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.group_conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output
    
    return x

class DepthSeparableConvBN(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(DepthSeparableConvBN, self).__init__()
        # groups = nInputPlane，就是Depthwise,如果groups = nInputPlane，kernel=(K, 1)（针对二维卷积，前面加上，groups=1 and kernel=(1, K)），就是可分离的。
        self.depth_wise = nn.Conv2d(in_dims, in_dims, kernel_size=3, padding=1, groups=in_dims)
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
        
        # 块通道数目，即分组数量，一般都是4
        bottleneck_channels = int(out_channels // 4)
        # 设置卷积通道数
        self.num_groups = num_groups
        # 分组卷积
        self.group_conv1 = GroupsConvBN(input_dims, bottleneck_channels, kernel=1, num_groups=num_groups)
        # 通道混洗
        # 深度可分离卷积
        self.depthwise_conv_bn = DepthSeparableConvBN(bottleneck_channels, bottleneck_channels)
        # 执行分组卷积
        self.group_conv2 = GroupsConvBN(bottleneck_channels, out_channels， kernel=1, num_groups=num_groups)
        # 最后激活层
        self.active = nn.ReLU(out_channels)
    def forward(self, input):
        out = self.group_conv1(input)
        out = channel_shuffle(out, self.num_groups)
        out = self.depthwise_conv_bn(out)
        out = self.group_conv2(out)
        out += input
        out = self.active(out)
        return out

        



def ShuffleNetUnitA(inputs, out_channels, num_groups):
    
    # 获取输入通道数
    in_channels = inputs.get_shape().as_list()[-1]
    # 获取输出通道数量
    #out_channels = in_channels
    # 块通道数目，即分组数量，一般都是4
    bottleneck_channels = out_channels // 4
    # 1*1 进行分组卷积
    x = group_conv(inputs, bottleneck_channels, kernel=1, strides=1, num_groups=num_groups)
    # 进行BN层
    x = slim.batch_norm(x)
    # 激活层
    x = tf.nn.relu(x)
    # 通道混洗
    x = channel_shuffle(x, num_groups)
    # 3*3 深度可分离卷积
    x = depthwise_conv_bn(x, bottleneck_channels, kernel_size=3, stride=1)
    # 1*1 分组卷积
    x = group_conv(x, out_channels, kernel=1, strides=1, num_groups=num_groups)
    # BN层
    x = slim.batch_norm(x)
    # 执行加法
    x = tf.keras.layers.add([inputs, x])
    # 激活层
    x = tf.nn.relu(x)
    
    return x

def ShuffleNetUnitB(inputs, out_channels, num_groups):
    """
    单元B，主要执行缩小一半的卷积操作
    Args:
        inputs ([type]): 输入数据
        out_channels ([type]): 输出数据
        num_groups ([type]): 卷积数量

    Returns:
        输出数据: 返回卷积结果值 
    """
    # 获取输入通道数目
    in_channels = inputs.get_shape().as_list()[-1]
    # 将输出通道数目减去输入通道数目？？？，需要其它的来进行池化
    out_channels -= in_channels
    # 计算每个块的大小
    bottleneck_channels = out_channels // 4
    # 进行分组卷积
    x = group_conv(inputs, bottleneck_channels, kernel=1, strides=1, num_groups=num_groups)
    x = slim.batch_norm(x)
    x = tf.nn.relu(x)
    # 通道混洗
    x = channel_shuffle(x, num_groups)
    # 进行深度可分离卷积
    x = depthwise_conv_bn(x, bottleneck_channels, kernel_size=3, stride=2)
    # 分组卷积
    x = group_conv(x, out_channels, kernel=1, strides=1, num_groups=num_groups)
    x = slim.batch_norm(x)
    # 进行均值池化
    y = slim.avg_pool2d(inputs, kernel_size=3, stride=2, padding='same')
    # 数据合并
    x = tf.concat([y, x], axis=-1)
    x = tf.nn.relu(x)
    
    return x

def ShuffleNetUnitV2A(inputs, out_channels, num_groups):
    """
    shufflenet v2 卷积单元1 主要负责直接卷积和输出
    注意输出通道数目为输入应该尽量相同

    https://zhuanlan.zhihu.com/p/48261931
    Args:
        inputs ([type]): 输入数据
        out_dims : 输出维度数据
        num_groups ([type]): 分组数量

    Returns:
        [type]: 返回的值
    """
    # 检查是否为2的倍数
    assert out_channels % 2 == 0
    # 先进行二维分离操作，将第三维度的分成两个部分
    shortcut, x = tf.split(inputs, 2, axis=-1)
    # 1*1 卷积
    x = conv(inputs, out_channels // 2, 1, 1, activation=True)
    # 深度可分离卷积
    x = depthwise_conv_bn(x, out_channels // 2, kernel_size=3, stride=1)
    # 1*1 卷积
    x = conv(x, out_channels // 2, 1, 1, activation=True)
    # 进行连接
    x = tf.concat([shortcut, x], axis=3)
    # 进行通道混洗
    x = channel_shuffle(x, 2)
    return x


def ShuffleNetUnitV2B(inputs, out_channels, num_groups):
    """
    单元B，主要执行缩小一半的卷积操作
    输出为输入的一倍
    Args:
        inputs ([type]): 输入数据
        out_channels ([type]): 输出数据
        num_groups ([type]): 卷积数量

    Returns:
        输出数据: 返回卷积结果值 
    """
    assert out_channels % 2 == 0
    # 进行双分支的数据复制
    shortcut = inputs
    # 获取输入通道数目
    in_channels = inputs.get_shape().as_list()[-1]
    # 进行1*1 卷积
    x = conv(inputs, out_channels // 2, 1, 1, activation=True)
    # 3*3 深度可分离卷积
    x = depthwise_conv_bn(x, out_channels // 2, kernel_size=3, stride=2)
    # 1*1 深度卷积
    x = conv(x, out_channels - in_channels, 1, 1, activation=True)
    # 深度可分离卷积
    shortcut = depthwise_conv_bn(shortcut, in_channels, kernel_size=3, stride= 2)
    # 1*1 卷积
    shortcut = conv(shortcut, in_channels, 1, 1, activation=True)
    # 进行组合
    output = tf.concat([shortcut, x], axis=-1)
    # 进行数据混合
    output = channel_shuffle(output, 2)
    return output

