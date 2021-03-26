<<<<<<< HEAD
import torch
import torch.nn as nn



class Conv2d(nn.Module):
    '''
    This class is for a convolutional layer.
    C
    '''
    def __init__(self, nIn, nOut, kSize, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        '''
        super().__init__()
        # 自适应边长填充
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=True)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output

class MaxPool2D(nn.Module):

    def __init__(self, nIn, nOut, kSize, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optinal stide for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1)/2)
        self.pool = nn.MaxPool2d(kSize, kSize, padding=padding)
    
    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.pool(input)
        return output

class MaxPool3D(nn.Module):

    def __init__(self, nIn, nOut, kSize, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optinal stide for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1)/2)
        self.pool = nn.MaxPool3d(kSize, padding=padding)
    
    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.pool(input)
        return output

class Conv2dBN(nn.Module):
    '''
       This class groups the convolution and batch normalization
       C+B
    '''
    def __init__(self, nIn, nOut, kSize, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optinal stide for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=True)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-05)

    def forward(self, input):
        '''

        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        output = self.bn(output)
        return output

class Conv2dBnRel(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    c+BN+RL
    '''
    def __init__(self, nIn, nOut, kSize, stride=1):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1)/2)
        #self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=True)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=True)
        #self.conv1 = nn.Conv2d(nOut, nOut, (1, kSize), stride=1, padding=(0, padding), bias=True)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-05)
        self.act = nn.ReLU()

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        #output = self.conv1(output)
        output = self.bn(output)
        output = self.act(output)
        return output



class BR2d(nn.Module):
    '''
        This class groups the batch normalization and PReLU activation
        BN+LU
    '''
    def __init__(self, nOut):
        '''
        :param nOut: output feature maps
        '''
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut, eps=1e-05)
        self.act = nn.ReLU()
        '''
        :return: normalized and thresholded feature map
        '''
        output = self.bn(input)
        output = self.act(output)
        return output


class Conv3d(nn.Module):
    '''
    This class is for a convolutional layer.
    3d卷积
    '''
    def __init__(self, nIn, nOut, kSize, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        pytorch 
        in N, Ci, D, H, W
        out N, Co, D, H, W
        tensorflow
        [batch, in_depth, in_height, in_width, in_channels] N,D,H,W,C
        '''
        super().__init__()
        # 自适应边长填充
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv3d(nIn, nOut, (kSize, kSize, kSize), stride=stride, padding=(padding, padding, padding), bias=True)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output

class Conv3dBN(nn.Module):
    '''
       This class groups the convolution and batch normalization
       C+B
    '''
    def __init__(self, nIn, nOut, kSize, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optinal stide for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv3d(nIn, nOut, (kSize, kSize, kSize), stride=stride, padding=(padding, padding, padding), bias=True)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-05)

    def forward(self, input):
        '''

        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        output = self.bn(output)
        return output

class Conv3dBnRel(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    c+BN+RL
    '''
    def __init__(self, nIn, nOut, kSize, stride=1):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv3d(nIn, nOut, (kSize, kSize, kSize), stride=stride, padding=(padding, padding, padding), bias=True)
        self.bn = nn.BatchNorm3d(nOut, eps=1e-05)
        self.act = nn.ReLU()


    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        #output = self.conv1(output)
        output = self.bn(output)
        output = self.act(output)
        return output

class BR3d(nn.Module):
    '''
        This class groups the batch normalization and PReLU activation
        BN+LU
    '''
    def __init__(self, nOut):
        '''
        :param nOut: output feature maps
        '''
        super().__init__()
        self.bn = nn.BatchNorm3d(nOut, eps=1e-05)
        self.act = nn.ReLU()

    def forward(self, input):
        '''
        :param input: input feature map
        :return: normalized and thresholded feature map
        '''
        output = self.bn(input)
        output = self.act(output)
        return output


class CDilated(nn.Module):
    '''
    This class defines the dilated convolution.
    空洞卷积
    '''
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        padding = int((kSize - 1)/2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=True, dilation=d)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output

class ResConv2d(nn.Module):
    def __init__(self, nIn, nOut, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        # 进行初始化
        super().__init__()
        if stride==1:
            self.conv1 = Conv2dBnRel(nIn, nOut, 3, 1)
            self.conv2 = Conv2dBnRel(nOut, nOut, 3, 1)
            self.conv3 = None
        else:
            self.conv1 = Conv2dBnRel(nIn, nOut, 3, 1)
            self.conv2 = Conv2dBnRel(nOut, nOut, 3, 2)
            self.conv3 = Conv2dBnRel(nIn, nOut, 1, 2)
    
            
    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        out = self.conv1(input)
        out = self.conv2(out)
        if self.conv3 is not None:
            residual = self.conv3(input)
            out += residual
        else:
            out += input
        return out


class ResConv3d(nn.Module):
    def __init__(self, nIn, nOut, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        # 进行初始化
        super().__init__()
        if stride==1:
            self.conv1 = Conv3dBnRel(nIn, nOut, 3, 1)
            self.conv2 = Conv3dBnRel(nOut, nOut, 3, 1)
            self.conv3 = None
        else:
            self.conv1 = Conv3dBnRel(nIn, nOut, 3, 1)
            self.conv2 = Conv3dBnRel(nOut, nOut, 3, 2)
            self.conv3 = Conv3dBnRel(nIn, nOut, 1, 2)
    
            
    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        out = self.conv1(input)
        out = self.conv2(out)
        if self.conv3 is not None:
            residual = self.conv3(input)
            out += residual
        else:
            out += input
        return out
        
class FastResConv2d(nn.Module):
    def __init__(self, nIn, nOut, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        # 进行初始化
        super().__init__()
        if stride==1:
            self.conv1 = Conv2dBnRel(nIn, nOut, 1, 1)
            self.conv2 = Conv2dBnRel(nOut, nOut, 3, 1)
            self.conv3 = Conv2dBnRel(nOut, nOut, 1, 1)
            self.conv4 = None
        else:
            self.conv1 = Conv2dBnRel(nIn, nOut, 1, 1)
            self.conv2 = Conv2dBnRel(nOut, nOut, 3, 2)
            self.conv3 = Conv2dBnRel(nOut, nOut, 1, 1)
            self.conv4 = Conv2dBnRel(nIn, nOut, 1, 2)
    
            
    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.conv4  is not None:
            residual = self.conv4(input)
            out += residual
        else:
            out += input
        return out


class FastResConv3d(nn.Module):
    def __init__(self, nIn, nOut, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        # 进行初始化
        super().__init__()
        if stride ==1:
            self.conv1 = Conv3dBnRel(nIn, nOut, 1, 1)
            self.conv2 = Conv3dBnRel(nOut, nOut, 3, 1)
            self.conv3 = Conv3dBnRel(nOut, nOut, 1, 1)
            self.conv4 = None
        else:
            self.conv1 = Conv3dBnRel(nIn, nOut, 1, 1)
            self.conv2 = Conv3dBnRel(nOut, nOut, 3, 2)
            self.conv3 = Conv3dBnRel(nOut, nOut, 1, 1)
            self.conv4 = Conv3dBnRel(nIn, nOut, 1, 2)
    
            
    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.conv4  is not None:
            residual = self.conv4(input)
            out += residual
        else:
            out += input
        return out
=======
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D
from tensorflow.keras.layers import MaxPool2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.layers import BatchNormalization, Activation


import numpy as np

slim = tf.contrib.slim

from core.config import cfg

# BN层和激活层，主要用于数据的筛选和归一化
def bnrelu(x):
    return tf.nn.relu(slim.batch_norm(x))

# 3d卷积层
def conv3d(x, dim, stride=1, bn=True):
    """
    3维度卷积和二维卷积基本相同，不做过多赘述
    Args:
        x ([type]): [description]
        dim ([type]): [description]
        stride (int, optional): [description]. Defaults to 1.
        bn (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    if bn:
        return slim.conv3d(bnrelu(x), dim, [3, 3, 3], stride=stride)
    else:
        return slim.conv3d(tf.nn.relu(x), dim, [3, 3, 3], stride=stride)

def conv3d_1x1(x, dim, stride=1, bn=True):
    if bn:
        return slim.conv3d(bnrelu(x), dim, [1, 1, 1], stride=stride)
    else:
        return slim.conv3d(tf.nn.relu(x), dim, [1, 1, 1], stride=stride)

# 二维卷积
def conv2d(x, dim, stride=1, bn=True):
    """二维卷积

    Args:
        x ([type]): 输入数据
        dim ([type]): 维度，一般为卷积核数量，一般为输出维度
        stride (int, optional): 步长. Defaults to 1.
        bn (bool, optional): 是否使用BN层. Defaults to True.

    Returns:
        [type]: slim.conv2d 默认pad模式是SAME ，卷积后的数据会变成和原来大小一样，比如n*m*w*h*c --> n*m*c*dim*h*w
    """
    if bn:
        return slim.conv2d(bnrelu(x), dim, [3, 3], stride=stride) # 注意这里指定的卷积核的大小都是3*3的大小，可以将其转变为两个3*1
    else:
        return slim.conv2d(tf.nn.relu(x), dim, [3, 3], stride=stride)
def conv2d_1x1(x, dim, stride=1, bn=True):
    # https://blog.csdn.net/weixin_44695969/article/details/102997574
    if bn:
        return slim.conv2d(bnrelu(x), dim, [1, 1], stride=stride) 
    else:
        return slim.conv2d(tf.nn.relu(x), dim, [1, 1], stride=stride)

def dilated_conv2d(x, dim, my_stride=1, my_rate=6, bn=True):
    """
    二维空洞卷积
    Args:
        x ([type]): [description]
        dim ([type]): [description]
        stride (int, optional): [description]. Defaults to 1.
        my_rate (int, optional): [description]. Defaults to 6.
        bn (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    x = slim.conv2d(x, dim, [3, 3], rate=my_rate, stride= my_stride, scope='rate_2d{}'.format(my_rate))
    if bn:
        x = slim.batch_norm(x)
        x = tf.nn.relu(x)
    else:
        x = tf.nn.relu(x)
    return x



def dilated_conv3d(x, dim, my_stride=1, my_rate=6, bn=True):
    if bn:
        return slim.conv3d(bnrelu(x), dim, [3, 3, 3], rate=my_rate, stride=my_stride, scope='3d_rate{}'.format(my_rate))
        #return slim.batch_norm(slim.conv3d(x), dim, [3, 3, 3], rate=my_rate, stride=my_stride, scope='3d_rate{}'.format(my_rate))

    else:
        return slim.conv3d(tf.nn.relu(x), dim, [3, 3, 3], rate=my_rate, stride=my_stride, scope='3d_rate{}'.format(my_rate))


def avg_pool(input_data, k_h, k_w, s_h, s_w, name, padding='SAME'):
        return tf.nn.avg_pool(input_data,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)


#def fast_shuffle_net_v2_conv2d(x,dim,stride=1):
def fast_res_conv2d(x,dim,stride=1):
    if stride==1:
        # y = conv2d_1x1(x,dim)
        # y = conv2d(x,dim)
        y = conv2d_1x1(x,dim)
    else:
        # y = conv2d_1x1(x,dim, stride=2)
        #y = conv2d(x, dim, stride=2)
        y = conv2d_1x1(x, dim, stride=2)
        # 使用单核进行卷积，大小和原来一样 将x进行转变
        x = slim.conv2d(x, dim, [1,1], stride=2)
    out = x + y # 将卷积数据进行叠加
    tf.add_to_collection("checkpoints", out) # 将数据放入集合参数
    return out

def fast_shuffle_net_v2_conv2d(x,dim,stride=1):
#def fast_res_conv2d(x,dim,stride=1):
    if stride==1:
        y = conv2d_1x1(x,dim)
        y = conv2d(y,dim)
        y = conv2d_1x1(y,dim)
    else:
        y = conv2d_1x1(x, dim, stride=1)
        y = conv2d(y, dim, stride=2)
        y = conv2d_1x1(y,dim, stride=1)
        # 使用单核进行卷积，大小和原来一样 将x进行转变
        x = slim.conv2d(x, dim, [1,1], stride=2)
    out = x + y # 将卷积数据进行叠加
    tf.add_to_collection("checkpoints", out) # 将数据放入集合参数
    return out

# 2维度卷积，主要是 (BN层->relu->slim.conv2d)*2；数据的输出是slim.conv2d(x)+conv2d(x),保证
def res_conv2d(x, dim, stride=1):
    """resnet的卷积层：
    
    详细见: https://blog.csdn.net/sunny_yeah_/article/details/89430124
    Args:
        x ([type]): [description]
        dim ([type]): [description]
        stride (int, optional): [description]. Defaults to 1.

    Returns:
        [type]: [description]
    """
    # 如果卷积步长为1则进行正常卷积
    if stride==1:
        y = conv2d(conv2d(x, dim), dim)
    else:
        # 这里是resnet卷积
        # 注意卷积步长为2时，长宽缩小为原来的一般
        # 否则将卷积步长设置为2，
        y = conv2d(conv2d(x, dim), dim, stride=2)
        # 使用单核进行卷积，大小和原来一样
        x = slim.conv2d(x, dim, [1,1], stride=2)

    out = x + y # 将卷积数据进行叠加
    tf.add_to_collection("checkpoints", out) # 将数据放入集合参数

    return out

def upnn3d(x, y, sc=2):
    # 获取最后一维数据长度，相当于通道数量，当前的通道数量
    dim = x.get_shape().as_list()[-1]
    # 获取相关维度
    bx, hx, wx, dx, _ = tf.unstack(tf.shape(x), num=5)
    # 获取目标相关维度
    by, hy, wy, dy, _ = tf.unstack(tf.shape(y), num=5)

    x1 = tf.reshape(tf.tile(x, [1, 1, sc, sc, sc]), [bx, sc*hx, sc*wx, sc*dx, dim])
    if not (sc*hx==hy and sc*wx==wy):
        x1 = x1[:, :hy, :wy]

    return x1
# 进行向上分解，主要进行上采样
def upnn2d(x, y, sc=2):
    dim = x.get_shape().as_list()[-1]
    bx, hx, wx, _ = tf.unstack(tf.shape(x), num=4)
    by, hy, wy, _ = tf.unstack(tf.shape(y), num=4)

    x1 = tf.reshape(
        tf.tile(x, [1,1,sc,sc]), # 进行数据扩张，将 最后两个维度扩充sc倍，一般为原来的2倍
        [bx, sc*hx, sc*wx, dim])
    if not (sc*hx==hy and sc*wx==wy):
        x1 = x1[:, :hy, :wy]

    return x1

def resize_like(inputs, ref):
    """
    主要进行线性插值缩放
    Args:
        inputs ([type]): [description]
        ref ([type]): [description]

    Returns:
        [type]: [description]
    """
    iH, iW = inputs.get_shape()[1], inputs.get_shape()[2]
    rH, rW = ref.get_shape()[1], ref.get_shape()[2]
    if iH == rH and iW == rW:
        return inputs
    return tf.image.resize_nearest_neighbor(inputs, [rH.value, rW.value])

def resize_depth(depth, dim, min_depth=0):
    depth = tf.image.resize_nearest_neighbor(depth[...,tf.newaxis], dim)
    if min_depth > 0:
        depth = tf.where(depth<min_depth, min_depth*tf.ones_like(depth), depth)

    return tf.squeeze(depth, axis=-1)

>>>>>>> test_run

