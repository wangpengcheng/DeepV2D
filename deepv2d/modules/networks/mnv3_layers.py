#coding:utf-8
#created by wangpengcheng
# https://blog.csdn.net/qq_37913573/article/details/90044592
# https://blog.csdn.net/weixin_44791964/article/details/104069624
import tensorflow as tf
import numpy as np
weight_decay=1e-5

def relu6(x, name='relu6'):
    """
    relu6激活函数
    """
    return tf.nn.relu6(x, name)

def hard_swish(x,name='hard_swish'):
    """
    h-swish 激活函数；h-swishj激活函数，代替swish函数，减少运算量，提高性能。
    Args:
        x ([type]): [description]
        name (str, optional): [description]. Defaults to 'hard_swish'.

    Returns:
        [type]: [description]
    """
    with tf.name_scope(name):
        h_swish = x*tf.nn.relu6(x+3)/6
        return h_swish

def batch_norm(x, momentum=0.997, epsilon=1e-3, train=True, name='bn'):
    """
    batch normal 标准层

    Args:
        x ([type]): [description]
        momentum (float, optional): [description]. Defaults to 0.997.
        epsilon ([type], optional): [description]. Defaults to 1e-3.
        train (bool, optional): [description]. Defaults to True.
        name (str, optional): [description]. Defaults to 'bn'.

    Returns:
        [type]: [description]
    """
    return tf.layers.batch_normalization(
                      x,
                      momentum=momentum,
                      epsilon=epsilon,
                      scale=True,
                      center=True,
                      training=train,
                      name=name)


def conv2d(
        input_, 
        output_dim, 
        k_h, 
        k_w, 
        d_h, 
        d_w, 
        stddev=0.09, 
        name='conv2d', 
        bias=False
        ):
    """
    卷积基本操作
    注意这里没有batch normal 层和relu激活层

    Args:
        input_ ([type]): 输入
        output_dim ([type]): 输出维度
        k_h ([type]): 卷积核高度
        k_w ([type]): 卷积核宽度
        d_h ([type]): h方向卷积步长
        d_w ([type]): w方向卷积步长
        stddev (float, optional): 初始化参数. Defaults to 0.09.
        name (str, optional): 卷基层名称. Defaults to 'conv2d'.
        bias (bool, optional): . Defaults to False.

    Returns:
        [type]: [description]
    """
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
              initializer=tf.truncated_normal_initializer(stddev=stddev))
        # 进行二维卷积
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        # 是否添加基础偏移
        if bias:
            biases = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)

        return conv


def conv2d_block(input, out_dim, k, s, is_train, name, h_swish=False):
    """
    一个卷积单元，也就是conv2d + batchnormalization + activation
    Args:
        input ([type]): [description]
        out_dim ([type]): [description]
        k ([type]): [description]
        s ([type]): [description]
        is_train (bool): [description]
        name ([type]): [description]
        h_swish (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    with tf.name_scope(name), tf.variable_scope(name):
        net = conv2d(input, out_dim, k, k, s, s, name='conv2d')
        net = batch_norm(net, train=is_train, name='bn')
        if h_swish == True:
            net = hard_swish(net)
        else:
            net = relu6(net)
        return net


def conv_1x1(input, output_dim, name, bias=False):
    """
    1x1 的逐点卷积
    Args:
        input ([type]): [description]
        output_dim ([type]): [description]
        name ([type]): [description]
        bias (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    with tf.name_scope(name):
        return conv2d(input, output_dim, 1, 1, 1, 1, stddev=0.09, name=name, bias=bias)

def pwise_block(input, output_dim, is_train, name, bias=False):
    """
    逐点卷积模块
    Args:
        input ([type]): [description]
        output_dim ([type]): [description]
        is_train (bool): [description]
        name ([type]): [description]
        bias (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    with tf.name_scope(name), tf.variable_scope(name):
        out=conv_1x1(input, output_dim, bias=bias, name='pwb')
        out=batch_norm(out, train=is_train, name='bn')
        out=relu6(out)
        return out

def dwise_conv(input, k_h=3, k_w=3, channel_multiplier= 1, strides=[1,1,1,1],
               padding='SAME', stddev=0.09, name='dwise_conv', bias=False):
    """
    深度可分离卷积
    Args:
        input ([type]): [description]
        k_h (int, optional): [description]. Defaults to 3.
        k_w (int, optional): [description]. Defaults to 3.
        channel_multiplier (int, optional): [description]. Defaults to 1.
        strides (list, optional): [description]. Defaults to [1,1,1,1].
        padding (str, optional): [description]. Defaults to 'SAME'.
        stddev (float, optional): [description]. Defaults to 0.09.
        name (str, optional): [description]. Defaults to 'dwise_conv'.
        bias (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    with tf.variable_scope(name):
        # 获取输出通道
        in_channel=input.get_shape().as_list()[-1]
        # 权重初始化
        w = tf.get_variable('w', [k_h, k_w, in_channel, channel_multiplier],
                        regularizer=tf.contrib.layers.l2_regularizer( weight_decay ),
                        initializer=tf.truncated_normal_initializer(stddev= stddev ))
        # 卷积
        conv = tf.nn.depthwise_conv2d(input, w, strides, padding, rate=None,name=None,data_format=None)
        if bias:
            biases = tf.get_variable('bias', [in_channel*channel_multiplier], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)

        return conv

def Fully_connected(x, units, layer_name='fully_connected') :
    with tf.name_scope(layer_name) :
        return tf.layers.dense(inputs=x, use_bias=True, units=units)

def global_avg(x,s=1):
    """
    全局平均池化层
    Args:
        x ([type]): [description]
        s (int, optional): [description]. Defaults to 1.

    Returns:
        [type]: [description]
    """
    with tf.name_scope('global_avg'):
        net = tf.layers.average_pooling2d(x, x.get_shape()[1:-1], s)
        return net


def hard_sigmoid(x, name='hard_sigmoid'):
    with tf.name_scope(name):
        h_sigmoid = tf.nn.relu6(x+3)/6
        return h_sigmoid

def conv2d_hs(input, output_dim, is_train, name, bias=False, se=False):
    """
    hs的conv模块
    Args:
        input ([type]): [description]
        output_dim ([type]): [description]
        is_train (bool): [description]
        name ([type]): [description]
        bias (bool, optional): [description]. Defaults to False.
        se (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    with tf.name_scope(name), tf.variable_scope(name):
        out = conv_1x1(input, output_dim, bias=bias, name='pwb')
        out = batch_norm(out, train=is_train, name='bn')
        out = hard_swish(out)
        # squeeze and excitation
        if se:
            channel = int(np.shape(out)[-1])
            out = squeeze_excitation_layer(out,out_dim=channel, ratio=4, layer_name='se_block')
        return out

def conv2d_NBN_hs(input, output_dim, name, bias=False):
    with tf.name_scope(name), tf.variable_scope(name):
        out=conv_1x1(input, output_dim, bias=bias, name='pwb')
        out=hard_swish(out)
        return out

def squeeze_excitation_layer(input, out_dim, ratio, layer_name):
    """
    注意力机制单元
    Args:
        input ([type]): [description]
        out_dim ([type]): [description]
        ratio ([type]): [description]
        layer_name ([type]): [description]

    Returns:
        [type]: [description]
    """
    with tf.name_scope(layer_name) :
        # 均值池化
        squeeze = global_avg(input)
        # 全连接层
        excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name+'_excitation1')
        # 激活层
        excitation = relu6(excitation)
        # 全连接层
        excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name+'_excitation2')
        # 激活层
        excitation = hard_sigmoid(excitation)
        # 输出维度
        excitation = tf.reshape(excitation, [-1,1,1,out_dim])
        # 在这里进行点成，添加概率
        scale = input * excitation

        return scale

def mnv3_block(input, k_s, expansion_ratio, output_dim, stride, is_train, name, bias=True, shortcut=True, h_swish=False, ratio=16, se=False):
    """
    mobilenet v3基础模块
    Args:
        input ([type]): 输入数据
        k_s ([type]): 卷积核的大小
        expansion_ratio ([type]): 扩展率
        output_dim ([type]): 输出维度
        stride ([type]): 
        is_train (bool): 是否训练
        name ([type]): 名称
        bias (bool, optional): 是否使用添加. Defaults to True.
        shortcut (bool, optional): [description]. Defaults to True.
        h_swish (bool, optional): [description]. Defaults to False.
        ratio (int, optional): [description]. Defaults to 16.
        se (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    with tf.name_scope(name), tf.variable_scope(name):
        # pw
        bottleneck_dim = expansion_ratio#round(expansion_ratio*input.get_shape().as_list()[-1])
        print(bottleneck_dim)
        # 1x1卷积调整通道数，通道数上升
        net = conv_1x1(input, bottleneck_dim, name='pw', bias=bias)
        # norm 标准层
        net = batch_norm(net, train=is_train, name='pw_bn')
        if h_swish:
            net = hard_swish(net)
        else:
            net = relu6(net)
        # 深度可分离卷积
        net = dwise_conv(net, k_w=k_s, k_h=k_s, strides=[1, stride, stride, 1], name='dw', bias=bias)
        # batch_norm 标准层
        net = batch_norm(net, train=is_train, name='dw_bn')
        # 是否使用高质量卷积
        if h_swish:
            net = hard_swish(net)
        else:
            net = relu6(net)
        # squeeze and excitation
        if se:
            # 获取通道数目
            channel = int(np.shape(net)[-1])
            # 注意力机制网络层
            net = squeeze_excitation_layer(net,out_dim=channel, ratio=ratio, layer_name='se_block')

        # pw & linear
        # 1x1卷积提升通道数目
        net = conv_1x1(net, output_dim, name='pw_linear', bias=bias)
        net = batch_norm(net, train=is_train, name='pw_linear_bn')

        # element wise add, only for stride==1
        if shortcut and stride == 1:
            # 进行添加工作
            in_dim = int(input.get_shape().as_list()[-1])
            # 
            net_dim = int(net.get_shape().as_list()[-1])
            if in_dim == net_dim:
                net += input
                net = tf.identity(net, name='output')

        return net


def flatten(x):
    #flattened=tf.reshape(input,[x.get_shape().as_list()[0], -1])  # or, tf.layers.flatten(x)
    return tf.contrib.layers.flatten(x)