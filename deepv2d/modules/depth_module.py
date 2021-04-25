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
class SoftArgmax(nn.Module):
    def __init__(self):
        super(SoftArgmax, self).__init__()

    def forward(self, depths, prob_volume):
        """ 
        Convert probability volume into point estimate of depth 
        转换概率体积为深度的点估计
        """
        # 特征概率  1 32 240 320 
        prob_volume = F.softmax(prob_volume, dim=1).permute(0, 2, 3, 1)
        #print(prob_volume.shape)
        # 计算特征概率合并图，1*480*640*1
        pred = torch.sum(depths*prob_volume, dim =-1)
        return pred # 返回深度估计值

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
        self.depths = torch.linspace(cfg.STRUCTURE.MIN_DEPTH, cfg.STRUCTURE.MAX_DEPTH, cfg.STRUCTURE.COST_VOLUME_DEPTH) # 进行线性插值获取深度序列
        self.pred_logits = []
        self.test_conv = Conv2d(3, 32, 3)
        # 注意这里默认的是线性插值
        self.transform = transforms.Compose([transforms.Resize(size=(self.ht, self.wd))])
        self.soft_argmax = SoftArgmax()
        self.EncoderFactory(cfg)
        self.DecoderFactory(cfg)
        self.StereoNetworkFactory(cfg)
    
    def stereo_network_avg(
        self, 
        Ts,
        images, 
        intrinsics
        ):
        """3D Matching Network with view pooling
        Ts: collection of pose estimates correponding to images   图片位姿集合
        images: rgb images      rgb图片 
        intrinsics: image intrinsics 相机内参
        adj_list: [n, m] matrix specifying frames co-visiblee frames 共同可见帧的矩阵
        """
        cfg = self.cfg
        # 进行线性插值，构造深度数据；用来随机初始化深度特征图
        depths = torch.linspace(cfg.STRUCTURE.MIN_DEPTH, cfg.STRUCTURE.MAX_DEPTH, cfg.STRUCTURE.COST_VOLUME_DEPTH, device=torch.device('cuda:0')) # 进行线性插值获取深度序列
        # 相机参数转换--将相机内参转换为四元组矩阵
        intrinsics = intrinsics_vec_to_matrix( intrinsics / 4.0 ) # 将相机参数转换为矩阵，并将其缩小为原来的一半
        # extract 2d feature maps from images and build cost volume # 进行编码，获取2d的图像信息
        # 进行图像编码，获取特征图 1*4*120*160*32
        # 在第5个通道上进行分离，获取数据
        batch, frames, channel, ht, wd = images.shape
        # 将其降低维度为4维 假设数据为1*4*480*640*3->4*480*640*3 方便卷积操作
        images = images.view([batch*frames, 3, ht, wd]) # 调整输入维度为图片数量*高*宽*3
        # 获取编码图片
        fmaps = self.encoder(images)
        fmaps = fmaps.view([batch, frames, 32, ht//8, wd//8]) # 1 4 32 30 40
        # #再重新调整顺序，还原维度信息
        # if self.cfg.STRUCTURE.ENCODER_MODE == 'resnet':
        #     fmaps = torch.reshape(fmaps, [batch, frames, 32, ht//4, wd//4]) # 1 4 32 60 80 
        # else:
        #     fmaps = torch.reshape(fmaps, [batch, frames, 32, ht//8, wd//8]) # 1 4 32 30 40 
        # #反投影，获取对应坐标对上的反向投影插值 1 4 30 40 32 64
        volume = operators.backproject_avg(Ts, depths, intrinsics, fmaps)
        pred = self.decoder(volume) # 1 32 240 320
        #self.pred_logits.append(torch.rand(1,32,240,320))
        # 返回最终产生的结果，可能存在多次三维金字塔卷积，取最后一次的结果
        #return self.soft_argmax(self.pred_logits[-1])
        return self.soft_argmax(depths, pred)
        
     
    def stereo_network_cat(
        self, 
        Ts, 
        images, 
        intrinsics
        ):
        """3D Matching Network with view pooling
        Ts: collection of pose estimates correponding to images   图片位姿集合
        images: rgb images      rgb图片 
        intrinsics: image intrinsics 相机内参
        adj_list: [n, m] matrix specifying frames co-visiblee frames 共同可见帧的矩阵
        """
        cfg = self.cfg
        # 进行线性插值，构造深度数据；用来随机初始化深度特征图
        depths = torch.linspace(cfg.STRUCTURE.MIN_DEPTH, cfg.STRUCTURE.MAX_DEPTH, cfg.STRUCTURE.COST_VOLUME_DEPTH, device=torch.device('cuda:0')) # 进行线性插值获取深度序列
        # 相机参数转换--将相机内参转换为四元组矩阵
        intrinsics = intrinsics_vec_to_matrix(intrinsics / 4.0) # 将相机参数转换为矩阵，并将其缩小为原来的一半
        # extract 2d feature maps from images and build cost volume # 进行编码，获取2d的图像信息
        # 进行图像编码，获取特征图 1*4*120*160*32
        # 在第5个通道上进行分离，获取数据
        batch, frames, channel, ht, wd = images.shape
        # 将其降低维度为4维 假设数据为1*4*480*640*3->4*480*640*3 方便卷积操作
        images = images.view([batch*frames, 3, ht, wd]) # 调整输入维度为图片数量*高*宽*3
        # 获取编码图片
        fmaps = self.encoder(images)
        fmaps = fmaps.view([batch, frames, 32, ht//8, wd//8]) # 1 4 32 30 40 
        # #再重新调整顺序，还原维度信息
        # if self.cfg.STRUCTURE.ENCODER_MODE == 'resnet':
        #     fmaps = torch.reshape(fmaps, [batch, frames, 32, ht//4, wd//4]) # 1 4 32 60 80 
        # else:
        #     fmaps = torch.reshape(fmaps, [batch, frames, 32, ht//8, wd//8]) # 1 4 32 30 40 
        # #反投影，获取对应坐标对上的反向投影插值 1 4 30 40 32 64
        volume = operators.backproject_cat(Ts, depths, intrinsics, fmaps)

        pred = self.decoder(volume) # 1 32 240 320
        #self.pred_logits.append(torch.rand(1,32,240,320))
        # 返回最终产生的结果，可能存在多次三维金字塔卷积，取最后一次的结果
        #return self.soft_argmax(self.pred_logits[-1])
        return self.soft_argmax(depths, pred)
    
    def StereoNetworkFactory(self, cfg):
        if self.cfg.STRUCTURE.MODE == 'avg':
            self.stereo_network = self.stereo_network_avg
        else:
            self.stereo_network = self.stereo_network_cat

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
        if self.cfg.STRUCTURE.MODE == 'avg':
            if self.cfg.STRUCTURE.DECODER_MODE == 'resnet':
                self.decoder = ResnetDecoder(64, self.pred_logits, (self.ht, self.wd), self.cfg.STRUCTURE.HG_COUNT, self.cfg.STRUCTURE.HG_DEPTH_COUNT)
            elif self.cfg.STRUCTURE.DECODER_MODE == 'fast_resnet':
                self.decoder = FastResnetDecoderAvg(64, self.pred_logits, (self.ht, self.wd), self.cfg.STRUCTURE.HG_COUNT, self.cfg.STRUCTURE.HG_DEPTH_COUNT)
            else:
                self.decoder = None
                print("cfg.FAST_MODE is error value:{}".format(self.cfg.FAST_MODE))
        else:
            if self.cfg.STRUCTURE.DECODER_MODE == 'resnet':
                self.decoder = ResnetDecoder(cfg.INPUT.FRAMES*32, self.pred_logits, (self.ht, self.wd), self.cfg.STRUCTURE.HG_COUNT, self.cfg.STRUCTURE.HG_DEPTH_COUNT)
            elif self.cfg.STRUCTURE.DECODER_MODE == 'fast_resnet':
                self.decoder = FastResnetDecoder(cfg.INPUT.FRAMES*32, self.pred_logits, (self.ht, self.wd), self.cfg.STRUCTURE.HG_COUNT, self.cfg.STRUCTURE.HG_DEPTH_COUNT)
            else:
                self.decoder = None
                print("cfg.FAST_MODE is error value:{}".format(self.cfg.FAST_MODE))

    def forward(self, poses, images, intrinsics, idx=None):
    #def forward(self, images, intrinsics, idx=None):
        # 映射图像数据
        images = 2 * (images / 255.0) - 1.0 # 将其映射到0-1
        # 获取宽度和高度
        ht = images.shape[-2]
        wd = images.shape[-1]
        self.input_dims = [ht, wd] # 获取输入信息
        spred = self.stereo_network(
            poses,
            images,
            intrinsics
        )
        return spred

    
   



