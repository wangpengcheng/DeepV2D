import tensorflow as tf
import numpy as np
slim = tf.contrib.slim

from .networks import hg
from .networks.layer_ops import *
# 导入mobile net模块
from .networks.mnv3_layers import *
from geometry.transformation import *
from geometry.intrinsics import *
from special_ops import operators

def print_net_shape():
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())  
        print(sess.run(tf.get_collection('temp_net'))) 

def add_depth_summaries(gt, pr):
    """添加深度信息

    Args:
        gt ([type]): [description]
        pr ([type]): [description]
    """
    gt = tf.reshape(gt, [-1])
    pr = tf.reshape(pr, [-1])
    # 筛选出非0.1~10.0的数据
    v = tf.where((gt>0.1) & (gt<10.0))
    # 获取筛选后的数据
    gt = tf.gather(gt, v)
    pr = tf.gather(pr, v)

    thresh = tf.maximum(gt / pr, pr / gt)
    delta = tf.reduce_mean(tf.to_float(thresh < 1.25))
    abs_rel = tf.reduce_mean(tf.abs(gt-pr) / gt)
    # 存储数据
    with tf.device('/cpu:0'):
        tf.summary.scalar("a1", delta)
        tf.summary.scalar("rel", abs_rel)


class DepthNetwork(object):
    def __init__(self, cfg, schedule=None, is_training=True, reuse=False):
        """
        深度估计网络，初始化参数
        Args:
            cfg ([type]): 配置类对象
            schedule ([type], optional): 训练进程调度器. Defaults to None.
            is_training (bool, optional): 是否正在训练，主要用分别，训练和推理两个阶段. Defaults to True.
            reuse (bool, optional): 是否重复. Defaults to False.
        """
        self.cfg = cfg
        self.reuse = reuse
        self.is_training = is_training
        self.schedule = schedule
        self.spreds = []
        self.summaries = {}
        # 在这里进行深度的初始化，主要是依靠线性插值
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

                
    def resnet_encoder(self, inputs, reuse=False):
        """ 2D feature extractor """
        # 在第5个通道上进行分离，获取数据
        batch, frames, ht, wd, _ = tf.unstack(tf.shape(inputs), num=5)
        # 将其降低维度为4维 假设数据为1*4*480*640*3->4*480*640*3
        inputs = tf.reshape(inputs, [batch*frames, ht, wd, 3]) # 调整输入维度为图片数量*高*宽*3

        with tf.variable_scope("encoder") as sc: #创建编码命名空间
            with slim.arg_scope([slim.batch_norm], **self.batch_norm_params):# 保存所有BN层的参数
                with slim.arg_scope([slim.conv2d], # 保存所有卷积层的参数
                                    weights_regularizer=slim.l2_regularizer(0.00005),
                                    normalizer_fn=None,
                                    activation_fn=None,
                                    reuse=reuse):
                    
                        # input 4*480*640*3
                        # 2d卷积网络 -- 这里可以拆成3个3*3的小网络，同时将输入图像betch更改为3
                        # 数据进行卷积，32*7*7的卷积，步长为2，大小将变为4*240*320*32
                        net = slim.conv2d(inputs, 32, [7, 7], stride=2)# slim.conv2d = cov2d+relu #1
                        # 卷积操作变为 4*240*320*32
                        net = res_conv2d(net, 32, 1) #2 
                        # 4*240*320*32
                        net = res_conv2d(net, 32, 1) #2 
                        # 4*240*320*32
                        net = res_conv2d(net, 32, 1) #2 
                        # 4*120*160*64
                        net = res_conv2d(net, 64, 2) #3 
                        # 4*160*120*64
                        net = res_conv2d(net, 64, 1) #2 
                        # 4*160*120*64
                        net = res_conv2d(net, 64, 1) #2 
                        # 4*160*120*64
                        net = res_conv2d(net, 64, 1) #2 
                        # 16层conv
                        for i in range(self.cfg.HG_2D_COUNT):
                            with tf.variable_scope("2d_hg1_%d"%i):
                                # 沙漏网络,4*120*160*128
                                net = hg.hourglass_2d(net, self.cfg.HG_2D_DEPTH_COUNT, 64)
                        # # 沙漏网络,4*120*160*128
                        # net = hg.hourglass_2d(net, 4, 64) # 52
                        # # 沙漏网络，4*120*160*64
                        # net = hg.hourglass_2d(net, 4, 64) # 52

                        # 卷积网络 4*120*160*32
                        embd = slim.conv2d(net, 32, [1, 1]) # 1
        # 重新进行缩放 1*4*120*160*32
        embd = tf.reshape(embd, [batch, frames, ht//4, wd//4, 32])
        return embd
    
    def fast_resnet_encoder(self, inputs, reuse=False):
        """
        使用1+3+1 的resnet卷积基本单元，每层网络深度加深1层，计算量减少一半左右
        Args:
            inputs ([type]): [description]
            reuse (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """
        """ 2D feature extractor """
        # 在第5个通道上进行分离，获取数据
        batch, frames, ht, wd, _ = tf.unstack(tf.shape(inputs), num=5)
        # 将其降低维度为4维 假设数据为1*4*480*640*3->4*480*640*3
        inputs = tf.reshape(inputs, [batch*frames, ht, wd, 3]) # 调整输入维度为图片数量*高*宽*3
        with tf.variable_scope("encoder") as sc: #创建编码命名空间
            with slim.arg_scope([slim.batch_norm], **self.batch_norm_params):# 保存所有BN层的参数
                with slim.arg_scope([slim.conv2d], # 保存所有卷积层的参数
                                    weights_regularizer=slim.l2_regularizer(0.00005),
                                    normalizer_fn=None,
                                    activation_fn=None,
                                    reuse=reuse):
                    # 进行卷积大小缩小一半
                    # 4*240*320*32
                    net = conv2d(inputs, 32, stride=2)
                    # 这里再次进行卷积，大小缩小一半
                    # 4*120*160*32
                    net = conv2d(inputs, 32, stride=2)
                    # 4*120*160*32
                    net = fast_res_conv2d(net, 32, 1) #2 
                    # 4*120*160*32
                    net = fast_res_conv2d(net, 32, 1) #2 
                    # 4*120*160*64
                    net = fast_res_conv2d(net, 64, 2) #3 
                    # 4*60*80*64
                    net = fast_res_conv2d(net, 64, 1) #2 
                    # 4*60*80*64
                    net = fast_res_conv2d(net, 64, 1) #2 
                    # 16层conv
                    for i in range(self.cfg.HG_2D_COUNT):
                        with tf.variable_scope("2d_hg1_%d"%i):
                            # 这里使用改进的快速2d沙漏网络
                            net = hg.fast_res_hourglass_2d(net, self.cfg.HG_2D_DEPTH_COUNT, 64)
                    # 卷积网络 4*60*80*32
                    embd = slim.conv2d(net, 32, [1, 1]) # 1
        # 重新进行缩放 1*4*120*160*32
        embd = tf.reshape(embd, [batch, frames, ht//8, wd//8, 32])
        return embd

    def mobilenet_encoder(self, inputs, reuse=False):
        """
        mobilenet 基本编码单元
        Args:
            inputs ([type]): [description]
            reuse (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """
        """ 2D feature extractor """
        # 在第5个通道上进行分离，获取数据
        batch, frames, ht, wd, _ = tf.unstack(tf.shape(inputs), num=5)
        # 将其降低维度为4维 假设数据为1*4*480*640*3->4*480*640*3
        inputs = tf.reshape(inputs, [batch*frames, ht, wd, 3]) # 调整输入维度为图片数量*高*宽*3

        with tf.variable_scope("encoder") as sc: #创建编码命名空间
            with slim.arg_scope([slim.batch_norm], **self.batch_norm_params):# 保存所有BN层的参数
                with slim.arg_scope([slim.conv2d], # 保存所有卷积层的参数
                                    weights_regularizer=slim.l2_regularizer(0.00005),
                                    normalizer_fn=None,
                                    activation_fn=None,
                                    reuse=reuse):
                    
                        # input 4*480*640*3
                        # 输入特征提取，尺度缩放为一半
                        net = conv2d_block(inputs, 16, 3, 2, is_train, name='conv1_1',h_swish=True)  # size/2
                        # 卷积操作变为 4*240*320*32
                        net = res_conv2d(net, 32, 1) #2 
                        # 4*240*320*32
                        net = res_conv2d(net, 32, 1) #2 
                        # 4*240*320*32
                        net = res_conv2d(net, 32, 1) #2 
                        # 4*120*160*64
                        net = res_conv2d(net, 64, 2) #3 
                        # 4*160*120*64
                        net = res_conv2d(net, 64, 1) #2 
                        # 4*160*120*64
                        net = res_conv2d(net, 64, 1) #2 
                        # 4*160*120*64
                        net = res_conv2d(net, 64, 1) #2 
                        # 16层conv
                        for i in range(self.cfg.HG_2D_COUNT):
                            with tf.variable_scope("2d_hg1_%d"%i):
                                # 沙漏网络,4*120*160*128
                                net = hg.hourglass_2d(net, self.cfg.HG_2D_DEPTH_COUNT, 64)
                        # # 沙漏网络,4*120*160*128
                        # net = hg.hourglass_2d(net, 4, 64) # 52
                        # # 沙漏网络，4*120*160*64
                        # net = hg.hourglass_2d(net, 4, 64) # 52

                        # 卷积网络 4*120*160*32
                        embd = slim.conv2d(net, 32, [1, 1]) # 1
        # 重新进行缩放 1*4*120*160*32
        embd = tf.reshape(embd, [batch, frames, ht//4, wd//4, 32])
        return embd


    # 编码部分，主要用来获取图像的2d特征信息
    def encoder(self, inputs, reuse=False):
        """
        编码器，根据配置文件执行对应的编码操作
        Args:
            inputs ([type]): 输入数据
            reuse (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """
        if self.cfg.ENCODER_MODE == 'resnet':
            return self.resnet_encoder(inputs, reuse)
        elif self.cfg.ENCODER_MODE == 'fast_resnet':
            return self.fast_resnet_encoder(inputs, reuse)
        elif self.cfg.ENCODER_MODE == 'mobilenet':
            return self.mobilenet_encoder(inputs, reuse)
        else:
            print("cfg.FAST_MODE is error value:{}".format(self.cfg.FAST_MODE)) 

    def resnet_decoder(self, volume):
        """
        后端解码模块
        Args:
            volume ([type]): decoder 主要特征部分
        """

        with slim.arg_scope([slim.batch_norm], **self.batch_norm_params):
            with slim.arg_scope([slim.conv3d],
                                weights_regularizer=slim.l2_regularizer(0.00005),
                                normalizer_fn=None,
                                activation_fn=None):
                # 获取数据维度batch ,frams,w,h,dim,6
                dim = tf.shape(volume)
                    # 将其维度进行强制转换 3*60*80*32*64
                volume = tf.reshape(volume, [dim[0]*dim[1], dim[2], dim[3], dim[4], 64])
                    # 进行三维特征卷积，卷积核大小为
                x = slim.conv3d(volume, 32, [1, 1, 1])
                    # 添加变量
                tf.add_to_collection("checkpoints", x)

                # multi-view convolution
                # 多视角卷积
                x = tf.add(x, conv3d(conv3d(x, 32), 32))
                # 重新整理输出为32维度
                x = tf.reshape(x, [dim[0], dim[1], dim[2], dim[3], dim[4], 32])
                # 沿着frame方向对所有帧求取平均值,1*120*160*32*32
                x = tf.reduce_mean(x, axis=1)
                tf.add_to_collection("checkpoints", x)
                self.pred_logits = []
                # 3维度特征金字塔提取
                for i in range(self.cfg.HG_COUNT):
                    with tf.variable_scope("hg1_%d"%i):
                        # 3d沙漏卷积，进行特征卷积，1*120*160*32*32
                        x = hg.hourglass_3d(x, self.cfg.HG_DEPTH_COUNT, 32)
                        # 将金字塔的结果进行输入
                        self.pred_logits.append(self.stereo_head(x))


    def fast_resnet_decoder(self, volume):
        """
        快速后端解码模块
        Args:
            volume ([type]): decoder 主要特征部分
        """

        with slim.arg_scope([slim.batch_norm], **self.batch_norm_params):
            with slim.arg_scope([slim.conv3d],
                                weights_regularizer=slim.l2_regularizer(0.00005),
                                normalizer_fn=None,
                                activation_fn=None):
                # 获取数据维度batch ,frams,w,h,dim,6
                dim = tf.shape(volume)
                    # 将其维度进行强制转换 3*60*80*32*64
                volume = tf.reshape(volume, [dim[0]*dim[1], dim[2], dim[3], dim[4], 64])
                    # 进行三维特征卷积，卷积核大小为
                x = slim.conv3d(volume, 32, [1, 1, 1])
                    # 添加变量
                tf.add_to_collection("checkpoints", x)

                # multi-view convolution
                # 多视角卷积
                x = tf.add(x, conv3d(x, 32))
                # 重新整理输出为32维度
                x = tf.reshape(x, [dim[0], dim[1], dim[2], dim[3], dim[4], 32])
                # 沿着frame方向对所有帧求取平均值,1*120*160*32*32
                x = tf.reduce_mean(x, axis=1)
                tf.add_to_collection("checkpoints", x)
                self.pred_logits = []
                # 3维度特征金字塔提取
                for i in range(self.cfg.HG_COUNT):
                    with tf.variable_scope("hg1_%d"%i):
                        # 3d沙漏卷积，进行特征卷积，1*120*160*32*32
                        x = hg.hourglass_3d(x, self.cfg.HG_DEPTH_COUNT, 32)
                        # 将金字塔的结果进行输入
                        self.pred_logits.append(self.fast_stereo_head(x))

            
    def decoder(self, volume):
        """
        后端解码模块
        Args:
            volume ([type]): decoder 主要特征部分
        """
        if self.cfg.DECODER_MODE == 'resnet':
            return self.resnet_decoder(volume)
        elif self.cfg.DECODER_MODE == 'fast_resnet':
            return self.fast_resnet_decoder(volume)
        elif self.cfg.DECODER_MODE == 'mobilenet':
            return self.mobilenet_decoder(volume)
        else:
            print("cfg.FAST_MODE is error value:{}".format(self.cfg.FAST_MODE)) 

    def stereo_head(self, x):
        """ Predict probability volume from hg features hg 的特征概率"""
        x = bnrelu(x)
        x = slim.conv3d(x, 32, [3, 3, 3], activation_fn=tf.nn.relu)
        x = slim.conv3d(x, 32, [3, 3, 3], activation_fn=tf.nn.relu)
        tf.add_to_collection("checkpoints", x)
        # 综合数据
        logits = slim.conv3d(x, 1, [1, 1, 1], activation_fn=None)
        logits = tf.squeeze(logits, axis=-1)
        # 根据输入进行线性插值
        logits = tf.image.resize_bilinear(logits, self.input_dims)
        return logits
    def fast_stereo_head(self, x):
        """ Predict probability volume from hg features hg 的特征概率"""
        x = bnrelu(x)
        x = slim.conv3d(x, 32, [3, 3, 3], activation_fn=tf.nn.relu)
        tf.add_to_collection("checkpoints", x)
        # 综合数据
        logits = slim.conv3d(x, 1, [1, 1, 1], activation_fn=None)
        logits = tf.squeeze(logits, axis=-1)
        # 根据输入进行线性插值
        logits = tf.image.resize_bilinear(logits, self.input_dims)
        return logits


    def soft_argmax(self, prob_volume):
        """ Convert probability volume into point estimate of depth 转换概率体积为深度的点估计"""
        # 特征概率 1*480*640*32
        prob_volume = tf.nn.softmax(prob_volume, axis=-1)
        # 计算特征概率合并图，1*480*640*32
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
        Ts: collection of pose estimates correponding to images   图片位姿集合
        images: rgb images      rgb图片 
        intrinsics: image intrinsics 相机内参
        adj_list: [n, m] matrix specifying frames co-visiblee frames 共同可见帧的矩阵
        """

        cfg = self.cfg
        # 进行线性插值，构造深度数据；用来随机初始化深度
        depths = tf.lin_space(cfg.MIN_DEPTH, cfg.MAX_DEPTH, cfg.COST_VOLUME_DEPTH) # 进行线性插值获取深度序列
        intrinsics = intrinsics_vec_to_matrix(intrinsics / 4.0) # 将相机参数转换为矩阵，并将其缩小为原来的一半
        # 使用stereo 命名
        with tf.variable_scope("stereo", reuse=self.reuse) as sc: 
            # extract 2d feature maps from images and build cost volume # 进行编码，获取2d的图像信息
            # 进行图像编码，获取特征图 1*4*120*160*32
            fmaps = self.encoder(images)
            # 反投影，获取对应坐标对上的反向投影插值
            volume = operators.backproject_avg(Ts, depths, intrinsics, fmaps, adj_list)
            # 进行编码模块
            self.decoder(volume)
        # 返回最终产生的结果
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
        print("ht:{}, wd: {}".format(ht,wd))
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
        # 遍历推理出来的深度图想
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
