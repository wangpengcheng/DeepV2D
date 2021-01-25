import tensorflow as tf
import numpy as np
slim = tf.contrib.slim

from .networks import hg
from .networks.layer_ops import *

from geometry.transformation import *
from geometry.intrinsics import *
from special_ops import operators

def add_depth_summaries(gt, pr):
    gt = tf.reshape(gt, [-1])
    pr = tf.reshape(pr, [-1])

    v = tf.where((gt>0.1) & (gt<10.0))
    gt = tf.gather(gt, v)
    pr = tf.gather(pr, v)

    thresh = tf.maximum(gt / pr, pr / gt)
    delta = tf.reduce_mean(tf.to_float(thresh < 1.25))
    abs_rel = tf.reduce_mean(tf.abs(gt-pr) / gt)

    with tf.device('/cpu:0'):
        tf.summary.scalar("a1", delta)
        tf.summary.scalar("rel", abs_rel)


class DepthNetwork(object):
    def __init__(self, cfg, schedule=None, is_training=True, reuse=False):
        self.cfg = cfg
        self.reuse = reuse
        self.is_training = is_training
        self.schedule = schedule

        self.summaries = {}
        self.depths = tf.lin_space(cfg.MIN_DEPTH, cfg.MAX_DEPTH, cfg.COST_VOLUME_DEPTH)

        self.batch_norm_params = {
          'decay': .995,
          'epsilon': 1e-5,
          'scale': True,
          'renorm': True,
          'renorm_clipping': schedule,
          'trainable': self.is_training,
          'is_training': self.is_training,
        }

    # 编码部分，主要用来获取图像的2d特征信息
    def encoder(self, inputs, reuse=False):
        """ 2D feature extractor """
        # 在第5个通道上进行分离
        batch, frames, ht, wd, _ = tf.unstack(tf.shape(inputs), num=5)
        inputs = tf.reshape(inputs, [batch*frames, ht, wd, 3]) # 调整输入维度为图片数量*高*宽*3

        with tf.variable_scope("encoder") as sc: 
            with slim.arg_scope([slim.batch_norm], **self.batch_norm_params):
                with slim.arg_scope([slim.conv2d],
                                    weights_regularizer=slim.l2_regularizer(0.00005),
                                    normalizer_fn=None,
                                    activation_fn=None,
                                    reuse=reuse):
                    # 2d卷积网络 -- 这里可以拆成3个3*3的小网络，同时将输入图像betch更改为3
                    net = slim.conv2d(inputs, 32, [7, 7], stride=2) # slim.conv2d = cov2d+relu

                    net = res_conv2d(net, 32, 1) # 卷积
                    net = res_conv2d(net, 32, 1)
                    net = res_conv2d(net, 32, 1)
                    net = res_conv2d(net, 64, 2)
                    net = res_conv2d(net, 64, 1)
                    net = res_conv2d(net, 64, 1)
                    net = res_conv2d(net, 64, 1)

                    net = hg.hourglass_2d(net, 4, 64)
                    net = hg.hourglass_2d(net, 4, 64)

                    embd = slim.conv2d(net, 32, [1, 1])

        embd = tf.reshape(embd, [batch, frames, ht//4, wd//4, 32])
        return embd

    def stereo_head(self, x):
        """ Predict probability volume from hg features hg的特征概率"""
        x = bnrelu(x)
        x = slim.conv3d(x, 32, [3, 3, 3], activation_fn=tf.nn.relu)
        x = slim.conv3d(x, 32, [3, 3, 3], activation_fn=tf.nn.relu)
        tf.add_to_collection("checkpoints", x)

        logits = slim.conv3d(x, 1, [1, 1, 1], activation_fn=None)
        logits = tf.squeeze(logits, axis=-1)
        # 日志
        logits = tf.image.resize_bilinear(logits, self.input_dims)
        return logits

    def soft_argmax(self, prob_volume):
        """ Convert probability volume into point estimate of depth 转换概率体积为深度的点估计"""
        prob_volume = tf.nn.softmax(prob_volume, axis=-1)
        pred = tf.reduce_sum(self.depths*prob_volume, axis= -1) # 对概率深度进行求和
        return pred # 返回深度估计值
    # 匹配网络avg方式下降
    def stereo_network_avg(self, 
        Ts, 
        images, 
        intrinsics, 
        adj_list=None
        ):
        """3D Matching Network with view pooling
        Ts: collection of pose estimates correponding to images  
        images: rgb images
        intrinsics: image intrinsics
        adj_list: [n, m] matrix specifying frames co-visiblee frames 共同可见帧的矩阵
        """

        cfg = self.cfg
        # 进行线性插值，构造深度数据
        depths = tf.lin_space(cfg.MIN_DEPTH, cfg.MAX_DEPTH, cfg.COST_VOLUME_DEPTH) # 进行线性插值获取深度序列
        intrinsics = intrinsics_vec_to_matrix(intrinsics / 4.0) # 将相机参数转换为矩阵

        with tf.variable_scope("stereo", reuse=self.reuse) as sc:
            # extract 2d feature maps from images and build cost volume # 进行编码，获取2d的图像信息
            # 进行图像编码
            fmaps = self.encoder(images)
            # 反投影
            volume = operators.backproject_avg(Ts, depths, intrinsics, fmaps, adj_list)

            self.spreds = []
            with slim.arg_scope([slim.batch_norm], **self.batch_norm_params):
                with slim.arg_scope([slim.conv3d],
                                    weights_regularizer=slim.l2_regularizer(0.00005),
                                    normalizer_fn=None,
                                    activation_fn=None):

                    dim = tf.shape(volume)
                    volume = tf.reshape(volume, [dim[0]*dim[1], dim[2], dim[3], dim[4], 64])

                    x = slim.conv3d(volume, 32, [1, 1, 1])
                    tf.add_to_collection("checkpoints", x)

                    # multi-view convolution
                    x = tf.add(x, conv3d(conv3d(x, 32), 32))

                    x = tf.reshape(x, [dim[0], dim[1], dim[2], dim[3], dim[4], 32])
                    x = tf.reduce_mean(x, axis=1)
                    tf.add_to_collection("checkpoints", x)

                    self.pred_logits = []
                    for i in range(self.cfg.HG_COUNT):
                        with tf.variable_scope("hg1_%d"%i):
                            x = hg.hourglass_3d(x, 4, 32)
                            self.pred_logits.append(self.stereo_head(x))

        return self.soft_argmax(self.pred_logits[-1])

    def stereo_network_cat(self, Ts, images, intrinsics):
        """3D Matching Network with view concatenation"""

        cfg = self.cfg
        depths = tf.lin_space(cfg.MIN_DEPTH, cfg.MAX_DEPTH, cfg.COST_VOLUME_DEPTH)
        intrinsics = intrinsics_vec_to_matrix(intrinsics / 4.0)

        with tf.variable_scope("stereo", reuse=self.reuse) as sc:
            # extract 2d feature maps from images and build cost volume
            fmaps = self.encoder(images)
            volume = operators.backproject_cat(Ts, depths, intrinsics, fmaps)

            self.spreds = []
            with slim.arg_scope([slim.batch_norm], **self.batch_norm_params):
                with slim.arg_scope([slim.conv3d],
                                    weights_regularizer=slim.l2_regularizer(0.00005),
                                    normalizer_fn=None,
                                    activation_fn=None):


                    x = slim.conv3d(volume, 48, [3, 3, 3])
                    x = tf.add(x, conv3d(conv3d(x, 48), 48))

                    self.pred_logits = []
                    for i in range(self.cfg.HG_COUNT):
                        with tf.variable_scope("hg1_%d"%i):
                            x = hg.hourglass_3d(x, 4, 48)
                            self.pred_logits.append(self.stereo_head(x))

        return self.soft_argmax(self.pred_logits[-1])


    def forward(self, poses, images, intrinsics, idx=None):

        images = 2 * (images / 255.0) - 1.0 # 将其映射到0-1
        
        ht = images.get_shape().as_list()[2] # 高度 
        wd = images.get_shape().as_list()[3] # 宽度
        
        self.input_dims = [ht, wd] # 获取输入信息

        # perform per-view average pooling 使用均值池化层模式
        if self.cfg.MODE == 'avg':
            spred = self.stereo_network_avg(poses, images, intrinsics, idx)

        # perform view concatenation 执行视图连接，连接位姿图像和参数
        elif self.cfg.MODE == 'concat':
            spred = self.stereo_network_cat(poses, images, intrinsics)
        # 返回最终的深度估计值
        return spred 

    # 计算loss
    def compute_loss(self, depth_gt, log_error=True):

        b_gt, h_gt, w_gt, _ = depth_gt.get_shape().as_list()

        total_loss = 0.0
        for i, logits in enumerate(self.pred_logits):

            pred = self.soft_argmax(logits)
            pred = tf.image.resize_bilinear(pred[...,tf.newaxis], [h_gt, w_gt])

            pred = tf.squeeze(pred, axis=-1)
            gt = tf.squeeze(depth_gt, axis=-1)

            valid = tf.to_float(gt>0.0)
            s = 1.0 / (tf.reduce_mean(valid) + 1e-8)

            gx = pred[:, :, 1:] - pred[:, :, :-1]
            gy = pred[:, 1:, :] - pred[:, :-1, :]
            vx = valid[:, :, 1:] * valid[:, :, :-1]
            vy = valid[:, 1:, :] * valid[:, :-1, :]

            # take l1 smoothness loss where gt depth is missing
            loss_smooth = \
                tf.reduce_mean((1-vx)*tf.abs(gx)) + \
                tf.reduce_mean((1-vy)*tf.abs(gy))

            loss_depth = s*tf.reduce_mean(valid*tf.abs(gt-pred))
            loss_i = self.cfg.TRAIN.SMOOTH_W * loss_smooth + loss_depth

            w = .5**(len(self.pred_logits)-i-1)
            total_loss += w * loss_i

        if log_error:
            add_depth_summaries(gt, pred)
        
        return total_loss
