import tensorflow as tf
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
# 二维卷积
def conv2d(x, dim, stride=1, bn=True):
    """[summary]

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
# 2维度卷积，主要是 (BN层->relu->slim.conv2d)*2；数据的输出是slim.conv2d(x)+conv2d(x),保证
def res_conv2d(x, dim, stride=1):
    # 如果卷积步长为1则进行正常卷积
    if stride==1:
        y = conv2d(conv2d(x, dim), dim)
    else:
        # 注意卷积步长为2时，长宽缩小为原来的一般
        # 否则将卷积步长设置为2，
        y = conv2d(conv2d(x, dim), dim, stride=2)
        # 使用单核进行卷积，大小和原来一样
        x = slim.conv2d(x, dim, [1,1], stride=2)

    out = x + y # 将卷积数据进行叠加
    tf.add_to_collection("checkpoints", out) # 将数据放入集合参数

    return out

def upnn3d(x, y, sc=2):
    # 获取最后一维数据长度，相当于通道数量
    dim = x.get_shape().as_list()[-1]
    # 
    bx, hx, wx, dx, _ = tf.unstack(tf.shape(x), num=5)
    by, hy, wy, dy, _ = tf.unstack(tf.shape(y), num=5)

    x1 = tf.reshape(tf.tile(x, [1,1,sc,sc,sc]), [bx, sc*hx, sc*wx, sc*dx, dim])
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
