from PIL import Image
import sys
import os
import urllib
import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
from tensorflow.python.platform import gfile

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  #selects a specific device


def get_trt_graph(
        pb_file ,
        output_name,
        precision_mode,
        batch_size=4,
        workspace_size=1 << 30
        ):
    # conver pb to FP32pb
    with tf.gfile.GFile(pb_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        print("load .pb")
    trt_graph = trt.create_inference_graph(
        input_graph_def=graph_def,
        outputs=[output_name],
        max_batch_size=batch_size,
        max_workspace_size_bytes=workspace_size,
        precision_mode=precision_mode)  # Get optimized graph
    print("create trt model done...")
    with gfile.FastGFile("model_tf_FP32.pb", 'wb') as f:
        f.write(trt_graph.SerializeToString())
        print("save TRTFP32.pb")
    return trt_graph


def get_tf_graph():
    with gfile.FastGFile(model_name, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        print("load .pb")
    return graph_def


if "__main__" in __name__:
    model_name = "deep_model.pb"
    #output_name = "softmax_out"
    output_name = "my_result"
    use_tensorrt = True
    precision_mode = "FP32"  #"FP16"
    batch_size = 1
    tf_config = tf.ConfigProto()
    print("[INFO] converting pb to FP32pb...")
    graph = get_trt_graph(model_name, output_name, precision_mode, batch_size)

