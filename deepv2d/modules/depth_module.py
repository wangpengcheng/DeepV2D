import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from .networks import hg
from .networks.layer_ops import *
from .networks.coders import *
from geometry.transformation import *
from geometry.intrinsics import *
from special_ops import operators

# def add_depth_summaries(gt, pr):
#     """
#     tensorborad 写入函数

#     Args:
#         gt ([type]): [description]
#         pr ([type]): [description]
#     """
#     gt = tf.reshape(gt, [-1])
#     pr = tf.reshape(pr, [-1])

#     v = tf.where((gt>0.1) & (gt<10.0))
#     gt = tf.gather(gt, v)
#     pr = tf.gather(pr, v)
#     # 筛选最大值
#     thresh = tf.maximum(gt / pr, pr / gt)
#     # 计算准确率平均值
#     delta = torch.mean(torch.FloatTensor(thresh < 1.25))
#     # 计算绝对值
#     abs_rel = torch.mean(torch.abs(gt-pr) / gt)

#     with tf.device('/cpu:0'):
#         # 写入相关数据
#         tf.summary.scalar("a1", delta)
#         tf.summary.scalar("rel", abs_rel)


class DepthModule(nn.Module):
    def __init__(self,  cfg, schedule=None, is_training=True, reuse=False):
        """
        深度网络基础类型

        Args:
            cfg ([type]): [description]
            schedule ([type], optional): [description]. Defaults to None.
            is_training (bool, optional): [description]. Defaults to True.
            reuse (bool, optional): [description]. Defaults to False.
        """
        super(DepthModule, self).__init__()
        self.cfg = cfg
        self.ht = int(cfg.INPUT.HEIGHT*cfg.INPUT.RESIZE)
        self.wd = int(cfg.INPUT.WIDTH*cfg.INPUT.RESIZE)
        self.pred_logits = []
        self.EncoderFactory(cfg)
        self.DecoderFactory(cfg)
    
    def soft_argmax(self, prob_volume):
        """ Convert probability volume into point estimate of depth 转换概率体积为深度的点估计"""
        # 特征概率 1*480*640*1
        prob_volume = F.softmax(prob_volume, dim=1)
        # 计算特征概率合并图，1*480*640*1
        pred = torch.sum(self.depths*prob_volume, axis= 1, name='my_result') # 对概率深度进行求和
        return pred # 返回深度估计值


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
        # 进行线性插值，构造深度数据；用来随机初始化深度特征图
        depths = torch.linspace(cfg.STRUCTURE.MIN_DEPTH, cfg.STRUCTURE.MAX_DEPTH, cfg.STRUCTURE.COST_VOLUME_DEPTH) # 进行线性插值获取深度序列
        # 相机参数转换
        intrinsics = intrinsics_vec_to_matrix(intrinsics / 4.0) # 将相机参数转换为矩阵，并将其缩小为原来的一半
        # extract 2d feature maps from images and build cost volume # 进行编码，获取2d的图像信息
        # 进行图像编码，获取特征图 1*4*120*160*32
         # 在第5个通道上进行分离，获取数据
        batch, frames, channel, ht, wd= images.shape
        # 将其降低维度为4维 假设数据为1*4*480*640*3->4*480*640*3
        images = torch.reshape(images, [batch*frames,3, ht, wd]) # 调整输入维度为图片数量*高*宽*3
        # 获取编码图片
        fmaps = self.encoder(images)
        # 再重新调整顺序
        if self.cfg.STRUCTURE.ENCODER_MODE == 'resnet':
            fmaps = torch.reshape(fmaps,[batch, frames, 32, ht//4, wd//4])
        else:
            fmaps = torch.reshape(fmaps,[batch, frames, 32, ht//8, wd//8])
        # 反投影，获取对应坐标对上的反向投影插值
        volume = operators.backproject_avg(Ts, depths, intrinsics, fmaps, adj_list)
        # 进行编码模块
        self.decoder(volume)
        # 返回最终产生的结果，可能存在多次三维金字塔卷积，取最后一次的结果
        return self.soft_argmax(self.pred_logits[-1])
     
    def stereo_network_cat(self, 
        Ts, 
        images, 
        intrinsics, 
        adj_list=None
        ):
        print("no !!!!")
    
    def EncoderFactory(self, cfg):
        """
        初始化编码器参数
        Args:
            cfg ([type]): [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.STRUCTURE.ENCODER_MODE == 'resnet':
            self.encoder = ResnetEncoder(3, 32, self.cfg.STRUCTURE.HG_2D_COUNT,self.cfg.STRUCTURE.HG_2D_DEPTH_COUNT)
        elif self.cfg.STRUCTURE.ENCODER_MODE == 'shufflenetv2':
            self.encoder = Shufflenetv2Encoder(3, 32, self.cfg.STRUCTURE.HG_2D_COUNT,self.cfg.STRUCTURE.HG_2D_DEPTH_COUNT)
        elif self.cfg.STRUCTURE.ENCODER_MODE == 'fast_resnet':
            self.encoder = FastResnetEncoder(3, 32, self.cfg.STRUCTURE.HG_2D_COUNT,self.cfg.STRUCTURE.HG_2D_DEPTH_COUNT)
        else:
            self.encoder = None
            print("cfg.FAST_MODE is error value:{}".format(self.cfg.FAST_MODE)) 
        
    def DecoderFactory(self, cfg):
        """
        
        初始化解码码器参数
        Args:
            cfg ([type]): [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.STRUCTURE.DECODER_MODE == 'resnet':
            self.decoder = ResnetDecoder(64, 1, self.pred_logits, (self.ht, self.wd), self.cfg.STRUCTURE.HG_COUNT, self.cfg.STRUCTURE.HG_DEPTH_COUNT)
        elif self.cfg.STRUCTURE.DECODER_MODE == 'fast_resnet':
            self.decoder = ResnetDecoder(64, 1, self.pred_logits, (self.ht, self.wd), self.cfg.STRUCTURE.HG_COUNT, self.cfg.STRUCTURE.HG_DEPTH_COUNT)
        else:
            self.decoder = None
            print("cfg.FAST_MODE is error value:{}".format(self.cfg.FAST_MODE))

    def forward(self, poses, images, intrinsics, idx=None):
        # 映射图像数据
        images = 2 * (images / 255.0) - 1.0 # 将其映射到0-1
        # 获取宽度和高度
        ht = images.shape[-2]
        wd = images.shape[-1]
        self.input_dims = [ht, wd] # 获取输入信息
        print("ht:{}, wd: {}".format(ht,wd))
        if self.cfg.STRUCTURE.MODE == 'avg':
            spred = self.stereo_network_avg(poses, images, intrinsics, idx)
        # perform view concatenation 执行视图连接，连接位姿图像和参数
        elif self.cfg.STRUCTURE.MODE == 'concat':
            spred = self.stereo_network_cat(poses, images, intrinsics)
        # 返回最终的深度估计值
        return spred

     # 计算loss
    def compute_loss(self, depth_gt, log_error=True):
        """[summary]

        Args:
            depth_gt ([type]): 真实深度值
            log_error (bool, optional): 是否输出日志. Defaults to True.

        Returns:
            [type]: 误差值
        """
        # 获取形状
        b_gt, h_gt, w_gt, _ = list(depth_gt.shape())

        total_loss = 0.0
        # 遍历推理出来的深度图想
        for i, logits in enumerate(self.pred_logits):
            # 进行归一化
            pred = self.soft_argmax(logits)
            # 将归一化的图像线性插值
            pred = nn.functional.interpolate(pred, size = (h_gt, w_gt),mode='bilinear')
            # 删除最后一个维度
            pred = torch.squeeze(pred, axis=-1)
            gt = torch.squeeze(depth_gt, axis=-1)
            # 所有深度大于0的值
            valid = torch.FloatTensor(gt>0.0)
            # 计算所有元素的均值
            s = 1.0 / (torch.mean(valid) + 1e-8)

            gx = pred[:, :, 1:] - pred[:, :, :-1]
            gy = pred[:, 1:, :] - pred[:, :-1, :]
            vx = valid[:, :, 1:] * valid[:, :, :-1]
            vy = valid[:, 1:, :] * valid[:, :-1, :]

            # take l1 smoothness loss where gt depth is missing
            loss_smooth = \
                torch.mean((1-vx)*torch.abs(gx)) + \
                torch.mean((1-vy)*torch.abs(gy))

            loss_depth = s*torch.mean(valid*torch.abs(gt-pred))
            loss_i = self.cfg.TRAIN.SMOOTH_W * loss_smooth + loss_depth

            w = .5**(len(self.pred_logits)-i-1)
            total_loss += w * loss_i

        # if log_error:
        #     #add_depth_summaries(gt, pred)
        
        return total_loss
   



