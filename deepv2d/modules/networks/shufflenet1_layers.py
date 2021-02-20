import tensorflow as tf
import numpy as np
slim = tf.contrib.slim
# https://blog.csdn.net/u014380165/article/details/81322175
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
    n, h, w, c = inputs.get_shape().as_list()
    # 重新进行交换操作
    x_reshaped = tf.reshape(inputs, [-1, h, w, num_groups, c // num_groups])
    # 进行更新
    x_transposed = tf.transpose(x_reshaped, [0, 1, 2, 4, 3])
    # 重设置形状
    output = tf.reshape(x_transposed, [-1, h, w, c])
    # 返回数据
    return output

def group_conv(inputs, filters, kernel, strides, num_groups):
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
    # 对输入数据进行分离
    conv_side_layers_tmp = tf.split(inputs, num_groups , axis=3)
    # 卷积结果数组
    conv_side_layers = []
    assert filters % num_groups == 0
    # 对每个进行卷积操作
    for layer in conv_side_layers_tmp:
        # 执行卷积操作
        temp = slim.conv2d(layer, filters//num_groups, kernel, strides, padding='same')
        # 添加操作
        conv_side_layers.append(temp)
    # 进行合并
    x = tf.concat(conv_side_layers, axis=-1)
    
    return x

def conv(inputs, filters, kernel_size, stride=1, activation=False):
    """
    标准卷积操作层
    Args:
        inputs ([type]): 输入数据
        filters ([type]): 输入维度大小
        kernel_size ([type]): 卷积核大小
        stride (int, optional): [description]. Defaults to 1.
        activation (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    # 进行卷积操作
    x = slim.conv2d(inputs, filters, kernel_size, stride)
    # 设置BN层
    x = slim.batch_norm(x)
    # 执行卷积操作
    if activation:
        x = tf.nn.relu(x)
        
    return x

def depthwise_conv_bn(inputs, out_dim, kernel_size, stride=1):
    """
    进行深度可分离卷积
    Args:
        inputs ([type]): 输入数据
        kernel_size ([type]): 卷积核大小
        stride (int, optional): 步长. Defaults to 1.

    Returns:
        Tensor : 最终返回结果 
    """
    x = slim.separable_conv2d(
        inputs,
        num_outputs = out_dim,
        kernel_size = kernel_size,
        stride = stride
        )
    x = slim.batch_norm(x)
        
    return x

def ShuffleNetUnitA(inputs, out_channels, num_groups):
    """
    shuffle net 卷积单元1
    Args:
        inputs ([type]): 输入数据
        out_dims : 输出维度数据
        num_groups ([type]): 分组数量

    Returns:
        [type]: 返回的值
    """
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


def FastShuffleNetv2Conv2d(x,dim,stride=1):
    if stride==1:
        y = conv(x, dim, 1, 1, activation=True)
        y = conv(x, dim, 3, 1, activation=True)
        y = conv(x, dim, 1, 1, activation=True)
    else:
        y = conv2d_1x1(x,dim, stride=2)
        y = conv2d(x,dim, stride=2)
        y = conv2d_1x1(x,dim, stride=2)
        y = conv(x, dim, 1, 1, activation=True)
        y = conv(x, dim, 3, 1, activation=True)
        y = conv(x, dim, 1, 1, activation=True)
        # 使用单核进行卷积，大小和原来一样 将x进行转变
        x = slim.conv2d(x, dim, [1,1], stride=2)
    out = x + y # 将卷积数据进行叠加
    tf.add_to_collection("checkpoints", out) # 将数据放入集合参数
    return out

def stage(inputs, out_channels, num_groups, n):
    """
    阶段模块
    Args:
        inputs ([type]): 输入数据
        out_channels ([type]): 输出通道数
        num_groups ([type]): 分组数据
        n ([type]): 单卷积数目

    Returns:
        [type]: [description]
    """
    
    x = ShuffleNetUnitB(inputs, out_channels, num_groups)
    # 执行第一个卷积
    for _ in range(n):
        out_channels = x.get_shape().as_list()[-1]
        x = ShuffleNetUnitA(x, out_channels, num_groups)
        
    return x

def stagev2(inputs, out_channels, num_groups, n):
    """
    阶段模块
    Args:
        inputs ([type]): 输入数据
        out_channels ([type]): 输出通道数
        num_groups ([type]): 分组数据
        n ([type]): 单卷积数目

    Returns:
        [type]: [description]
    """
    
    x = ShuffleNetUnitV2B(inputs, out_channels, num_groups)
    # 执行第一个卷积
    for _ in range(n):
        out_channels = x.get_shape().as_list()[-1]
        x = ShuffleNetUnitV2A(x, out_channels, num_groups)
        
    return x

def ShuffleNet(inputs, first_stage_channels, num_groups):
    # 进行二维卷积
    x = tf.keras.layers.Conv2D(filters=24, 
                               kernel_size=3, 
                               strides=2, 
                               padding='same')(inputs)
    #最大池化
    x = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
    # 进行卷积操作，附带的单层卷积数量为1
    x = stage(x, first_stage_channels, num_groups, n=3)
    # 进行卷积操作，进行维度升级
    x = stage(x, first_stage_channels*2, num_groups, n=7)
    # 进行维度审计
    x = stage(x, first_stage_channels*4, num_groups, n=3)
    # 进行全局池化操作
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # 执行全连接层
    x = tf.keras.layers.Dense(1000)(x)
    
    return x

def ShuffleNetV2(inputs, first_stage_channels, num_groups):
    # 进行二维卷积
    x = tf.keras.layers.Conv2D(filters=24, 
                               kernel_size=3, 
                               strides=2, 
                               padding='same')(inputs)
    #最大池化
    x = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
    # 进行卷积操作，附带的单层卷积数量为1
    x = stagev2(x, first_stage_channels, num_groups, n=3)
    # 进行卷积操作，进行维度升级
    x = stagev2(x, first_stage_channels*2, num_groups, n=7)
    # 进行维度审计
    x = stagev2(x, first_stage_channels*4, num_groups, n=3)
    # 进行全局池化操作
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # 执行全连接层
    x = tf.keras.layers.Dense(1000)(x)
    
    return x

# inputs = np.zeros((1, 224, 224, 3), np.float32)
# a = ShuffleNetV2(inputs, 144, 1)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(a))