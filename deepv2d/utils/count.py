# -*- coding:UTF-8 -*-

import tensorflow as tf

def count_param():       # 计算网络参数量
    total_parameters = 0
    for v in tf.trainable_variables():
        shape = v.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print('网络总参数量：', total_parameters)

def count_flops(graph):
    """输出网络的flops
    graph =tf.get_default_graph()
    Args:
        graph ([graph]): 模型参数，注意一定要 在sess.run(tf.global_variables_initializer())后加入
    """
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    print('FLOPs: {}'.format(flops.total_float_ops))


def stats_graph(graph):
    """输入参数量和flops

    Args:
        graph ([type]): [description]
    """
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('\n FLOPs: {};    Trainable params: {} \n'.format(flops.total_float_ops, params.total_parameters))
