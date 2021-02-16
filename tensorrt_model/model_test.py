import tensorflow as tf
import os

model_names = '/home/ls/workspace_2021_01_17/DeepV2D/tensorrt_model/DeepNet.pb'
'''
把pb文件的节点读出来
'''
def print_tensors(pb_file):
    print('Model File: {}\n'.format(pb_file))
    # read pb into graph_def
    with tf.gfile.GFile(pb_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # import graph_def
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)

    # print operations
    for op in graph.get_operations():
        print(op.name + '\t' + str(op.values()))

def inference():
    
    with tf.gfile.FastGFile(model_names, 'rb') as model_file:
        graph = tf.Graph()
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(model_file.read())
        images = tf.random_normal([1,5,240,320,3])
        [output_image] = tf.import_graph_def(graph_def,
                          input_map={'input_image': images},
                          return_elements=['output_label:0'],
                          name='output')
        sess = tf.Session()
        label = sess.run(output_image)
        return label


print_tensors(model_names)