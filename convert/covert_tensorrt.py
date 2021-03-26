import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

input_saved_model_dir = "/home/ls/workspace_2021_01_17/DeepV2D/tensorrt_model"
output_saved_model_dir = './tensorrtmodels/'

# converter = trt.TrtGraphConverter(
#     input_saved_model_dir=input_saved_model_dir,
#     precision_mode="INT8",
#     maximum_cached_engines=100)

# converter.convert()
# converter.save(output_saved_model_dir)

# input_data = tf.random_normal([1,5,240,320,3], minval=0.0, maxval = 255.0)
# with tf.Session() as sess:
#     # First load the SavedModel into the session    
#     tf.saved_model.loader.load(
#         sess, [tf.saved_model.tag_constants.SERVING],
#        output_saved_model_dir)
    
#     output = sess.run([output_tensor], feed_dict={input_tensor: input_data})


# with tf.Session() as sess:
#     # First create a `Saver` object (for saving and rebuilding a
#     # model) and import your `MetaGraphDef` protocol buffer into it:
#     saver = tf.train.import_meta_graph("/home/ls/workspace_2021_01_17/DeepV2D/checkpoints/tum/tmu_model/_stage_2.ckpt.meta")
#     # Then restore your training data from checkpoint files:
#     saver.restore(sess, "/home/ls/workspace_2021_01_17/DeepV2D/checkpoints/tum/tmu_model/_stage_2.ckpt")
#     # Finally, freeze the graph:
#     frozen_graph = tf.graph_util.convert_variables_to_constants(
#         sess,
#         tf.get_default_graph().as_graph_def(),
#         output_node_names=['my_result'])

# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.python.compiler.tensorrt import trt_convert as trt
params = trt.DEFAULT_TRT_CONVERSION_PARAMS
params._replace(precision_mode=trt.TrtPrecisionMode.FP32)
converter = trt.TrtGraphConverterV2(input_saved_model_dir="./model/yolo_tf_model", conversion_params=params)
# 完成转换,但是此时没有进行优化,优化在执行推理时完成
converter.convert()
converter.save('./model/trt_savedmodel')