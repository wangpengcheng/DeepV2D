import numpy as np
from utils.einsum import einsum
import torch

MIN_DEPTH = 0.1
# 按照形状进行网格化
def coords_grid(shape, homogeneous=True):
    """ grid of pixel coordinates 获取每个像素的网格坐标点"""
    yy,xx= torch.meshgrid(torch.range(0,shape[-2]-1),torch.range(0,shape[-1]-1))
    xx = xx.float()
    yy = yy.float()
    if homogeneous:
        coords = torch.stack((xx, yy, torch.ones_like(xx)), dim=-1)
    else:
        coords = torch.stack((xx, yy), dim=-1)
    new_shape = np.ones_like(list(shape)[:-2])
    new_shape = np.concatenate([new_shape,list(shape)[-2:], [-1]], axis=0).tolist()
    coords = torch.reshape(coords, new_shape)
    tile = np.concatenate([list(shape)[:-2], [1, 1, 1]], axis=0).tolist() # 获取小块
    coords = torch.Tensor.repeat(coords, tile) # 对坐标点张量进行扩张
    return coords

def extract_and_reshape_intrinsics(intrinsics, shape=None):
    """ Extracts (fx, fy, cx, cy) from intrinsics matrix """

    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]

    if shape is not None:
        batch = list((fx.shape)[:1])
        fillr = np.ones_like(list(shape)[1:])
        k_shape = np.concatenate([batch,fillr],axis=0).tolist()
        fx = torch.reshape(fx, k_shape)
        fy = torch.reshape(fy, k_shape)
        cx = torch.reshape(cx, k_shape)
        cy = torch.reshape(cy, k_shape)

    return (fx, fy, cx, cy)

# 将depthmap转换为点云,主要是根据相机参数还原原始的3D点云
def backproject(depth, intrinsics, jacobian=False):
    """ backproject depth map to point cloud """

    coords = coords_grid(depth.shape, homogeneous=True)
    x, y, _ = torch.unbind(coords, dim=-1)

    x_shape = x.shape
    x_shape = torch.IntTensor(list(x_shape))
    fx, fy, cx, cy = extract_and_reshape_intrinsics(intrinsics, x_shape)
    # 在这里矫正fx
    Z = depth.clone() # 获取全像素的真实深度
    X = Z * (x - cx) / fx # 获取x坐标
    Y = Z * (y - cy) / fy
    points = torch.stack([X, Y, Z], dim=-1)

    if jacobian:
        o = torch.zeros_like(Z) # used to fill in zeros

        # jacobian w.r.t (fx, fy)
        jacobian_intrinsics = torch.stack([
            torch.stack([-X / fx], dim=-1),
            torch.stack([-Y / fy], dim=-1),
            torch.stack([o], dim=-1),
            torch.stack([o], dim=-1)], dim=-2)
        return points, jacobian_intrinsics
    return points


def project(points, intrinsics, jacobian=False):
    
    """ project point cloud onto image 将点云投影到图像上""" 
    X, Y, Z = torch.unbind(points,dim=-1)
    Z[ Z< MIN_DEPTH ]=MIN_DEPTH
    #Z = torch.max(Z, MIN_DEPTH) # 获取最大深度，主要这里需要设置一下最小深度
    

    x_shape = X.shape # 获取x数据的长度
    fx, fy, cx, cy = extract_and_reshape_intrinsics(intrinsics, x_shape) # 调整相机内参矩阵

    x = fx * (X / Z) + cx
    y = fy * (Y / Z) + cy
    coords = torch.stack([x, y], dim=-1)

    if jacobian:
        o = torch.zeros_like(x) # used to fill in zeros
        zinv1 = torch.nonzero(Z <= MIN_DEPTH+.01, torch.zeros_like(Z), 1.0 / Z)
        zinv2 = torch.nonzero(Z <= MIN_DEPTH+.01, torch.zeros_like(Z), 1.0 / Z**2)

        # jacobian w.r.t (X, Y, Z)
        jacobian_points = torch.stack([
            torch.stack([fx * zinv1, o, -fx * X * zinv2], dim=-1),
            torch.stack([o, fy * zinv1, -fy * Y * zinv2], dim=-1)], dim=-2)

        # jacobian w.r.t (fx, fy)
        jacobian_intrinsics = torch.stack([
            torch.stack([X * zinv1], dim=-1),
            torch.stack([Y * zinv1], dim=-1),], dim=-2)

        return coords, (jacobian_points, jacobian_intrinsics)

    return coords
