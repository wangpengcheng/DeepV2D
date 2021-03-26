import tensorflow as tf
from tensorflow.python.framework import graph_util

def ckpt2pb():
    with tf.Graph().as_default() as graph_old:
        isess = tf.InteractiveSession()

        ckpt_filename = '/home/ls/workspace_2021_01_17/DeepV2D/checkpoints/tum/tmu_model'
        isess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph(ckpt_filename+'.meta', clear_devices=True)
        saver.restore(isess, ckpt_filename)

        constant_graph = graph_util.convert_variables_to_constants(isess, isess.graph_def, ["Cls/fc/biases"])
        constant_graph = graph_util.remove_training_nodes(constant_graph)

        with tf.gfile.GFile('/home/ls/workspace_2021_01_17/DeepV2D/tensorrt_model/model.pb', mode='wb') as f:
            f.write(constant_graph.SerializeToString())