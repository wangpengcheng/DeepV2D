
import tensorflow as tf
import os

def set_gpus(cfg):
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    # 定义使用的GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.TRAIN.USE_GPU
    # 定义TensorFlow配置
    config = tf.ConfigProto()
    # 配置GPU内存分配方式，按需增长，很关键
    config.gpu_options.allow_growth = True
    # 配置可使用的显存比例，为所有显存的80%
    config.gpu_options.per_process_gpu_memory_fraction = cfg.TRAIN.GPU_MEMORY
    # 在创建session的时候把config作为参数传进去
    sess = tf.InteractiveSession(config = config)

    return sess

    