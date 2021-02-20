import tensorflow as tf
import numpy as np
slim = tf.contrib.slim
from .shufflenet1_layers import *
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

def fast_res_hourglass_2d(x, n, dim, expand=64):
    """
    快速沙漏网络，去除了一个映射通道
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
        low2 = fast_res_hourglass_2d(low1, n-1, dim2) # 注意这里返回值和low相似，是x的一半
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
    return out


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


# 3d 沙漏网络
def fast_hourglass_3d_01(x, n, dim, expand=48):
    dim2 = dim + expand
    ax = conv3d_1x1(x, dim)
    ax = conv3d(ax, dim)
    x = x + conv3d_1x1(ax, dim)
    tf.add_to_collection("checkpoints", x)

    # 线性层缩放减半
    pool1 = slim.max_pool3d(x, [2, 2, 2], padding='SAME')
    #pool1 = conv3d_1x1(pool1, dim2)
    low1 = conv3d(pool1, dim2)
    if n>1:
        low2 = fast_hourglass_3d(low1, n-1, dim2)
    else:
        low2 = low1 + conv3d(low1, dim2)

    low3 = conv3d_1x1(low2, dim)
    up2 = upnn3d(low3, x)
    out = up2 + x
    tf.add_to_collection("checkpoints", out)

    return out
# 快速3d沙漏网络
def fast_hourglass_3d(x, n, dim, expand=48):
    dim2 = dim + expand

    x = x + conv3d_1x1(x, dim)
    tf.add_to_collection("checkpoints", x)

    # 线性层缩放减半
    pool1 = slim.max_pool3d(x, [2, 2, 2], padding='SAME')

    low1 = conv3d(pool1, dim2)
    if n>1:
        low2 = fast_hourglass_3d(low1, n-1, dim2)
    else:
        low2 = low1 + conv3d(low1, dim2)


    low3 = conv3d_1x1(low2, dim)
    up2 = upnn3d(low3, x)
    out = up2 + x
    tf.add_to_collection("checkpoints", out)

    return out

def aspp_2d(net, dim, expand=64):
    """
    aspp尺度卷积网络，主要是为了抵抗尺度变换

    Args:
        net ([type]): [description]
        dim ([type]): [description]
        expand (int, optional): [description]. Defaults to 64.

    Returns:
        [type]: [description]
    """
    out_dim = dim + expand

    with tf.variable_scope('aspp2d', [net]) as sc:
        aspp_list = []
        # 进行特征卷积
        # net1 = conv2d_1x1(net, dim)
        # 先进行1*1卷积
        # branch_1 = conv2d_1x1(net, out_dim)
        # 进行均值池化
        # branch_1 = slim.avg_pool2d(net,out_dim,1, padding='SAME',scope = 'aspp_2d_avg_pool')
        #branch_1 = avg_pool(net, 1, 1, 1, 1,)
        # 进行二维卷积1*1卷积
        branch_1 = slim.conv2d(net, out_dim, [1, 1], stride=1, scope='1x1conv')
        #tf.add_to_collection("checkpoints", branch_1)
        # 将其添加到
        aspp_list.append(branch_1)
        # 进行空洞卷积
        for i in range(3):
            branch_2 = slim.conv2d(net, out_dim, [3, 3], stride=1, rate=6*(i+1), scope='rate{}'.format(6*(i+1)))
            #branch_2 = dilated_conv2d(net, out_dim, stride=1, my_rate=6*(i+1))
            aspp_list.append(branch_2)
        # 进行维度合并
        temp_concat = tf.concat(aspp_list, 1)
        # out =net1 + conv2d_1x1(temp_concat, dim)
        out = conv(temp_concat, dim, 1, activation=True)
        #tf.add_to_collection("checkpoints", out)
        return out 

def aspp_3d(net, dim, expand=48):
    out_dim = dim + expand
    with tf.variable_scope('aspp3d', [net]) as sc:
        aspp_list = []
        # 进行二维卷积1*1卷积
        # branch_1 = slim.conv3d(net, out_dim, [1, 1, 1], stride=1, scope='1x1x1conv')
        # 进行二维卷积
        # net = net + conv3d_1x1(net,dim)

        # tf.add_to_collection("checkpoints", net)
        # 进行均值池化,再进行卷积,注意这里是没有进行扩充
        pool1 = slim.avg_pool3d(net,[2,2,2],padding='SAME', scope = 'aspp_3d_avg_pool')

        # 进行上采样
        low1 = conv3d(pool1,out_dim)
        
        # 添加第一个卷积
        aspp_list.append(conv3d_1x1(pool1, out_dim))

        # 进行空洞卷积
        for i in range(3):
            branch_2 = dilated_conv3d(low1, out_dim, my_rate = 6*(i+ 1))
            aspp_list.append(branch_2)

        # 进行维度合并
        temp_concat = tf.concat(aspp_list, 1)
        # 进行多维卷积
        low3 = conv3d_1x1(temp_concat, dim)

        up2 = upnn3d(low3, net)
        out = up2 + net

        tf.add_to_collection("checkpoints", out)
    return out
        