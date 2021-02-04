import tensorflow as tf
import numpy as np
slim = tf.contrib.slim

from .layer_ops import *
    
# 沙漏网络2d
# 这里分为了两个分支网络，在两次卷积之后进入分支道路，进行卷积，主要应该是为了抗拒，尺度变换，估计抗模糊能力有限
def hourglass_2d(x, n, dim, expand=64):
    
    # 获取输出维度64+64
    dim2 = dim + expand # 没迭代一次维度增加64
    # 进行两次二维卷积 大小不变，维度变为64；注意这里的残差特性

    # 第一级残差网络
    x = x + conv2d(conv2d(x, dim), dim)
    #进行池化,主要是为了抗拒过敏；输出大小变为原来的一半4*60*80*64
    pool1 = slim.max_pool2d(x, [2, 2], padding='SAME')
    # 再进行卷积，大小不变维度变为128,生成low1 4*60*80*128
    low1 = conv2d(pool1, dim2)
    # 重复进行卷积，
    if n>1:
        low2 = hourglass_2d(low1, n-1, dim2) # 注意这里返回值和low相似，是x的一半
    else:
        low2 = conv2d(low1, dim2)
    # 输出维度(h-2*n)(w-2*n)*128
    # 再次进行卷积4*60*80*64
    low3 = conv2d(low2, dim)
    # 对卷积结果进上采样，保持和x原来一样的维度,主要是对池化后的数据进行重新填充
    up2 = upnn2d(low3, x) # 进行扩充，值为原来的一倍
    # 将获取的其它维度的信息进行叠加，主要是缩放8之后的特征叠加
    # 2级残差
    out = up2 + x
    tf.add_to_collection("checkpoints", out)

    return out

def fast_hourglass_2d(x, n, dim, expand=64):
    """
    快速沙漏网络
    Args:
        x ([type]): [description]
        n ([type]): [description]
        dim ([type]): [description]
        expand (int, optional): [description]. Defaults to 64.

    Returns:
        [type]: [description]
    """
     # 获取输出维度64+64
    dim2 = dim + expand # 没迭代一次维度增加64
    # 进行两次二维卷积 大小不变，维度变为64；注意这里的残差特性

    # 使用1x1的卷积

    x = x + conv2d_1x1(x, dim)

    #进行池化,主要是为了抗拒过敏；输出大小变为原来的一半4*60*80*64
    pool1 = slim.max_pool2d(x, [2, 2], padding='SAME')
    # 再进行卷积，大小不变维度变为128,生成low1 4*60*80*128
    low1 = conv2d(pool1, dim2)
    # 重复进行卷积，
    if n>1:
        low2 = fast_hourglass_2d(low1, n-1, dim2) # 注意这里返回值和low相似，是x的一半
    else:
        low2 = conv2d(low1, dim2)

    # 再次进行卷积4*60*80*64
    #low3 = conv2d(low2, dim)
    low3 = conv2d_1x1(low2, dim)
    # 对卷积结果进上采样，保持和x原来一样的维度,主要是对池化后的数据进行重新填充
    up2 = upnn2d(low3, x) # 进行扩充，值为原来的一倍
    # 将获取的其它维度的信息进行叠加，主要是缩放8之后的特征叠加
    # 2级残差
    out = up2 + x
    tf.add_to_collection("checkpoints", out)


# 3d 沙漏网络
def hourglass_3d(x, n, dim, expand=48):
    dim2 = dim + expand

    x = x + conv3d(conv3d(x, dim), dim)
    tf.add_to_collection("checkpoints", x)

    pool1 = slim.max_pool3d(x, [2, 2, 2], padding='SAME')

    low1 = conv3d(pool1, dim2)
    if n>1:
        low2 = hourglass_3d(low1, n-1, dim2)
    else:
        low2 = low1 + conv3d(conv3d(low1, dim2), dim2)

    low3 = conv3d(low2, dim)
    up2 = upnn3d(low3, x)

    out = up2 + x
    tf.add_to_collection("checkpoints", out)

    return out
