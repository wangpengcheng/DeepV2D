
import torchvision.transforms as transforms
import torch 
import numpy as np
from utils.einsum import einsum
# 将相机参数转换为矩阵
def intrinsics_vec_to_matrix(kvec):
    # 将相机参数进行拼接
    fx, fy, cx, cy = torch.unbind(kvec, dim=-1)
    z = torch.zeros_like(fx) #零阶矩阵
    o = torch.ones_like(fx) #1阶矩阵

    K = torch.stack([fx, z, cx, z, fy, cy, z, z, o], dim=-1)
    K = torch.reshape(K, kvec.shape[:-1] + tuple([3, 3])) # 相机矩阵拼接
    return K

def intrinsics_matrix_to_vec(kmat):
    fx = kmat[..., 0, 0]
    fy = kmat[..., 1, 1]
    cx = kmat[..., 0, 2]
    cy = kmat[..., 1, 2]
    return torch.stack([fx, fy, cx, cy], dim=-1) # 提取所有帧的相机内参，并转化为一维度数据

def update_intrinsics(intrinsics, delta_focal):
    # 将位姿信息转转为kvect
    kvec = intrinsics_matrix_to_vec(intrinsics)
    fx, fy, cx, cy = torch.unbind(kvec, num=4, dim=-1)
    df = torch.squeeze(delta_focal, -1)

    # update the focal lengths
    fx = torch.exp(df) * fx
    fy = torch.exp(df) * fy

    kvec = torch.stack([fx, fy, cx, cy], dim=-1)
    kmat = intrinsics_vec_to_matrix(kvec)
    return kmat
# 对深度进行重新缩放
def rescale_depth(depth, downscale=4):
    depth = torch.unsqueeze(depth, dim=0) # 将所有维度扩展
    j,n,w,h= depth.shape[:]
    w = int(w/downscale+0.5)
    h = int(h/downscale+0.5)
    depth = torch.nn.functional.interpolate(depth,size=[w, h],mode="nearest")
    return torch.squeeze(depth, dim=0) # 对其进行维度压缩

def rescale_depth_and_intrinsics(depth, intrinsics, downscale=4):
    sc = torch.tensor([1.0/downscale, 1.0/downscale, 1.0], dtype=torch.float32) # 构建缩放矩阵
    intrinsics = einsum('...ij,i->...ij', intrinsics, sc) # 求点乘将值缩放为原来的1/4
    depth = rescale_depth(depth, downscale=downscale) #
    return depth, intrinsics

def rescale_depths_and_intrinsics(depth, intrinsics, downscale=4):
    batch, frames, height, width = [depth.shape[i] for i in range(4)] # 获取数据维度信息
    depth = torch.reshape(depth, [batch*frames, height, width]) # 将深度图，转换为三维的叠加产物，注意这里维度缩减了
    depth, intrinsics = rescale_depth_and_intrinsics(depth, intrinsics, downscale)
    new_shape = torch.IntTensor(list(depth.shape[1:]))
    new_dim = torch.IntTensor([batch,frames])
    depth = torch.reshape(depth,
        list(torch.cat((new_dim,new_shape), dim=0)))
    return depth, intrinsics
