import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile


'''
将节点名字打印出来
'''
def getAllNodes(checkpoint_path):
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    # Print tensor name and values
    for key in var_to_shape_map:
        print("tensor_name: ", key)
        #print(reader.get_tensor(key))

def dispTensorboard(file_path,out_file_path):
    graph = tf.get_default_graph()
    graphdef = graph.as_graph_def()
    _ = tf.train.import_meta_graph(file_path + '.meta', clear_devices=True)
    summary_write = tf.summary.FileWriter(out_file_path , graph)


def freeze_graph(ckpt, output_graph):
    #输出节点的名称，最直观的是从tensorboard里读，一般就是最后输出的节点，例如这里就是输出accuracy的节点
    output_node_names = 'my_result'
    # 开始节点 split:0 结束节点 Sum:0
    # saver = tf.train.import_meta_graph(ckpt+'.meta', clear_devices=True)
    saver = tf.compat.v1.train.import_meta_graph(ckpt + '.meta')
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt)
        # for node in sess.graph_def.node:
        #     print(node)
        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=input_graph_def,
            output_node_names=output_node_names.split(',')
        )
        with tf.gfile.GFile(output_graph, 'wb') as fw:
            fw.write(output_graph_def.SerializeToString())
        print('{} ops in the final graph.'.format(len(output_graph_def.node)))

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

'''
从ckpt中读取图结构，输出可以被tensorboard读取的图文件
'''
def showNetFromCkpt(path):
    
    graph = tf.get_default_graph()
    graphdef = graph.as_graph_def()
    _ = tf.train.import_meta_graph(path)
    #tensorboard的图文件输出的位置
    #使用tensorboard --logdir=E:\\MachineLearningProjects\\ViolentDetection_JD\\savedModels\\graph 进入tensorboard
    summary_write = tf.summary.FileWriter("E:\\MachineLearningProjects\\ViolentDetection_JD\\savedModels", graph)
    summary_write.close()

if __name__ == '__main__':
    #注意这里的path必须是绝对路径！！
    # 获取文件路径
    ckpt_path='/home/ls/workspace_2021_01_17/DeepV2D/checkpoints/tum/tmu_model/_stage_2.ckpt'

    #读取图文件，读完了就注释了就行，把输出节点写到上面的freeze_graph函数里
    #showNetFromCkpt(ckpt_path+".meta")
    #getAllNodes(ckpt_path)

    #将ckpt转换为pb,这里写pb的路径，也必须是绝对路径
    output_graph_path='/home/ls/workspace_2021_01_17/DeepV2D/tensorrt_model/DeepNet.pb'
    # 开始进行
    freeze_graph(ckpt_path, output_graph_path)

    # #将Pb文件的节点打印出来，看看有没有问题
    print_tensors(output_graph_path)
    #dispTensorboard(ckpt_path,"/home/ls/workspace_2021_01_17/DeepV2D/log/test")


