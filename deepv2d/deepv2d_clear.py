import tensorflow as tf
import numpy as np
import time
import cv2
import vis
from scipy import interpolate
import matplotlib.pyplot as plt
from tensorflow.python.client import timeline
from modules.depth import DepthNetwork
from tensorflow.python.framework import graph_util
from fcrn import fcrn
from geometry.intrinsics import *
from geometry.transformation import *
from geometry import projective_ops

# 填充深度图像，主要是对深度图像进行线性插值
def fill_depth(depth):
    """ Fill in the holes in the depth map  将整个图填充为深度图像
    """
    ht, wd = depth.shape
    x, y = np.meshgrid(np.arange(wd), np.arange(ht))
    xx = x[depth > 0].astype(np.float32)
    yy = y[depth > 0].astype(np.float32)
    zz = depth[depth > 0].ravel()
    # 进行插值 
    return interpolate.griddata((xx, yy), zz, (x, y), method='linear')

# 将旋转转换为角度
def vee(R):
    x1 = R[2,1] - R[1,2]
    x2 = R[0,2] - R[2,0]
    x3 = R[1,0] - R[0,1]
    return np.array([x1, x2, x3])

# 获取位姿距离；主要用来表示变换的剧烈程度
def pose_distance(G):
    # 获取相机旋转R和平移t 
    R, t = G[:3,:3], G[:3,3]
    r = vee(R)
    # dr的均方和
    dR = np.sqrt(np.sum(r**2))
    dt = np.sqrt(np.sum(t**2))
    return dR, dt


class DeepV2D:
    """
    推理测试主要类
    """
    def __init__(self, 
                 cfg,
                 ckpt,
                 image_dims=None,
                 mode='keyframe'
                 ):

        self.cfg = cfg # cfg配置文件
        self.ckpt = ckpt # ckpt文件位置
        self.mode = mode # 加载的模型文件

        if image_dims is not None:
            self.image_dims = image_dims
        else:
            if cfg.STRUCTURE.MODE == 'concat':
                self.image_dims = [cfg.INPUT.FRAMES, cfg.INPUT.HEIGHT, cfg.INPUT.WIDTH]
            else:
                self.image_dims = [cfg.INPUT.FRAMES, int(cfg.INPUT.HEIGHT*cfg.INPUT.RESIZE), int(cfg.INPUT.WIDTH*cfg.INPUT.RESIZE)] # 构建输入的训练图片维度

        self.outputs = {}
        # 创建预定于变量
        self._create_placeholders()
        # 创建深度网络
        self._build_depth_graph()
        self.depths = []

        # 加载模型
        self.saver = tf.train.Saver(tf.model_variables()) #构建存储模型

    # 创建session
    def set_session(self, sess):
        self.sess = sess
        # 初始化所有变脸
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # 进行模型恢复
        self.saver.restore(self.sess, self.ckpt)


    def _create_placeholders(self):
        """
        创建预定义变量；主要是构建数据
        """
        frames, ht, wd = self.image_dims
        # 输入数据占位，注意这里的输入是3张图片
        self.images_placeholder = tf.placeholder(tf.float32, [frames, ht, wd, 3], name="images")
        # 如果是关键帧模式frames值为1
        if self.mode == 'keyframe':
            self.depths_placeholder = tf.placeholder(tf.float32, [1, ht, wd],name="depths")
        else:
            self.depths_placeholder = tf.placeholder(tf.float32, [frames, ht, wd], name="depths")
        # 位姿内存分配；每一个位姿都是4*4的一个矩阵
        self.poses_placeholder = tf.placeholder(tf.float32, [frames, 4, 4], name="poses")
        # 相机内参矩阵一维数组
        self.intrinsics_placeholder = tf.placeholder(tf.float32, [4], name="intrinsics")
    
    def _build_depth_graph(self):
        """
        构建深度图信息
        """
        self.depth_net = DepthNetwork(self.cfg.STRUCTURE, is_training=False)
        images = self.images_placeholder[tf.newaxis]
        poses = self.poses_placeholder[tf.newaxis]
        intrinsics = self.intrinsics_placeholder[tf.newaxis]

        # convert pose matrix into SE3 object
        # 将位姿矩阵转换为se矩阵
        Ts = VideoSE3Transformation( matrix = poses)


        depths = self.depth_net.forward(
            Ts, # 相机位姿 
            images, # 图像 
            intrinsics
            )
        # 更新深度图
        self.outputs['depths'] = depths

    def inference(
        self, 
        images, 
        poses, 
        intrinsics=None, 
        iters=2, 
        i_step=-1
        ):
        # 图片
        self.images = images
        # 位姿
        self.poses = poses
        # 内参
        self.intrinsics = intrinsics
        # 进行数据绑定
        feed_dict = {
                self.images_placeholder: self.images,
                self.poses_placeholder: self.poses,
                self.intrinsics_placeholder: self.intrinsics
            }
        # 设置推理时间统计
        if i_step >= 0:
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            # 开始执行推理,并采集参数
            self.depths = self.sess.run(
                self.outputs['depths'], 
                feed_dict=feed_dict,
                options=options,
                run_metadata=run_metadata
                )
            # 将使用历史，保存为json文件
            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            # 写入文件夹
            with open('./log/time_lines/timeline_02_step_%d.json' %i_step, 'w') as f:
                    f.write(chrome_trace)
        else:
            self.depths = self.sess.run(
                self.outputs['depths'], 
                feed_dict=feed_dict
                )
        
        return self.depths

    def toPb(self, pb_name):
        """
        将模型存储为pb文件
        Args:
            pb_name ([type]): [description]
        """
        # 将节点作为输出
        output_graph_def = graph_util.convert_variables_to_constants(self.sess, self.sess.graph_def, ["my_result"])
        output_graph_def = graph_util.remove_training_nodes(output_graph_def)
        # 
        with tf.gfile.GFile(pb_name, "wb") as f: #保存模型
            f.write(output_graph_def.SerializeToString()) #序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node)) #得到当前图有几个操作节点