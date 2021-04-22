import numpy as np
from utils.einsum import einsum
import torch

MIN_DEPTH = 0.1
# 按照形状进行网格化
def coords_grid(shape, homogeneous=True):
    """ grid of pixel coordinates 获取每个像素的网格坐标点"""
    b, n, c, h, w = shape[:]
    # 进行平滑操作
    yy, xx= torch.meshgrid(torch.arange(h), torch.arange(w))
    xx = xx.float().cuda()
    yy = yy.float().cuda()
    # 进行维度拼接
    coords = torch.stack((xx, yy), dim=-1)
    #coords = torch.stack((xx, yy, torch.ones_like(xx)), dim=-1)
    coords = coords.repeat(b, n, c, 1, 1, 1) # 对坐标点张量进行扩张
    return coords

def extract_and_reshape_intrinsics(intrinsics, shape=None):
    """ Extracts (fx, fy, cx, cy) from intrinsics matrix 
        更改fx,fy,cx,,cy 相关系数
    """
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    if shape is not None:
        batch = fx.shape[0]
        b, n, c ,h,w = shape[:]
        fx = fx.reshape(batch,1,1,1,1).repeat(1,n,c,h,w)
        fy = fy.reshape(batch,1,1,1,1).repeat(1,n,c,h,w)
        cx = cx.reshape(batch,1,1,1,1).repeat(1,n,c,h,w)
        cy = cy.reshape(batch,1,1,1,1).repeat(1,n,c,h,w)
    return (fx, fy, cx, cy)

# 将depthmap转换为点云,主要是根据相机参数还原原始的3D点云
def backproject(depth, intrinsics, jacobian=False):
    """ backproject depth map to point cloud """
    # s
    coords = coords_grid(depth.shape, homogeneous=True)
    #coords = coords.cuda()
    x, y = torch.unbind(coords, dim=-1)

    x_shape = x.shape
    x_shape = list(x_shape)
    fx, fy, cx, cy = extract_and_reshape_intrinsics(intrinsics, x_shape)
    # 在这里矫正fx
    Z = depth.clone() # 获取全像素的真实深度
    X = Z * (x - cx) / fx # 获取x坐标
    # 计算Y值
    Y = Z * (y - cy) / fy
    points = torch.stack([X, Y, Z], dim=-1)
    return points


def project(points, intrinsics):
    """ project point cloud onto image 将点云投影到图像上""" 
    # 获取点云图
    X, Y, Z = torch.unbind(points, dim=-1)
    # 设置最小深度
    Z[ Z < MIN_DEPTH]=MIN_DEPTH
    
    x_shape = X.shape # 获取x数据的长度
    x_shape = list(x_shape)
    fx, fy, cx, cy = extract_and_reshape_intrinsics(intrinsics, x_shape) # 调整相机内参矩阵
    # 计算对应的像素点位置坐标
    x = fx * (X / Z) + cx
    y = fy * (Y / Z) + cy
    coords = torch.stack([x, y], dim=-1)
    return coords
