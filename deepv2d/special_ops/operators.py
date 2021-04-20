import torch
import os.path as osp
import numpy as np
from geometry.projective_ops import *
from utils.bilinear_sampler import *
import torch.nn.functional as F
#cholesky_solve = cholesky.solve
def my_gather(input, indexs, dim=1):
    return input.index_select(dim, indexs)


def adj_to_inds(num=-1, adj_list=None):
    """ Convert adjency list into list of edge indicies (ii, jj) = (from, to)
        将邻接列表转换为边缘索引列表
    """
    if adj_list is None:
        ii, jj = torch.meshgrid(torch.arange(1), torch.arange(0, num)) # 进行平滑操作
    else:
        n, m = torch.unbind(torch.Tensor(list(adj_list.shape), num=2))
        ii,jj = torch.split(adj_list, [1, m-1], dim=-1)
        ii = ii.repeat(1, m-1)
    ii = torch.reshape(ii, [-1])
    jj = torch.reshape(jj, [-1])
    
    return ii, jj

def get_cood(depths, intrinsics, Tij):

    batch, num, c, ht, wd = depths.shape[:]
    pt= backproject(depths, intrinsics)
    # 进行向量分解出
    X, Y, Z = torch.unbind(pt, dim=-1)
    zero = torch.ones_like(X)
    # 进行合并
    PT = torch.stack([X, Y, Z, zero], dim = -1)
    # 对Tij进行整合,方便进行乘法
    Tij = torch.reshape(Tij, (batch, num, 1, 1, 1, 4, 4))

    Tij = Tij.repeat(1, 1, c, ht, wd, 1, 1)
    Tij = Tij.view(batch*num*c*ht*wd, 4, 4)
    PT = PT.view(batch*num*c*ht*wd, 4, 1)
    # 进行乘法运算
    re = torch.bmm(Tij, PT)
    re = re.view(batch, num, c, ht, wd, 4)

    # 对最后一个维度进行分解
    X, Y, Z, one = torch.unbind(re, dim=-1)
    pt1 = torch.stack([X, Y, Z], dim = -1)
    coords = project(pt1, intrinsics)

    return coords 

def TS_inverse(pose):
    """
    位姿矩阵的逆矩阵
    Args:
        pose ([type]): [description]
    """
    b, n, w, h = pose.shape[:]
    #print(type(b))
    # 提取列数据
    col1, col2, col3, col4 = torch.unbind(pose, dim=-1)
    # 对每列数据进行分解
    a, b, c, zero = torch.chunk(col1, 4, dim=-1)
    d, e, f, _ = torch.chunk(col2, 4, dim=-1)
    g, h, i, _ = torch.chunk(col3, 4, dim=-1)
    x, y, z, one = torch.chunk(col4, 4, dim=-1)
    res = torch.cat([
        a, b, c, z*c-x*a-y*b,
        d, e, f, z*f-x*d-y*e,
        g, h, i, z*i-x*g-y*h,
        zero, zero, zero, one
        ], dim = -1)
    #print(res.shape)
    res = res.view(pose.shape)
    return  res

def backproject_avg(
            Ts, 
            depths, 
            intrinsics, 
            fmaps
            ):
    """

    Args:
        Ts ([type]): 相机位姿集合
        depths ([type]): 深度图像集合
        intrinsics ([type]): 相机内参
        fmaps ([type]): 特征图
        adj_list ([type], optional): 调整序列. Defaults to None.

    Returns:
        [type]: [description]
    """
    # use_cuda_backproject
    use_cuda_backproject = False
    # 获取通道数目
    dim = fmaps.shape[2]
    # 获取深度数量
    dd = depths.shape[0]
    # 将特征图进行矩阵分解，获取batch、num、ht和wd等，
    batch, num, c , ht, wd = fmaps.shape[0:] # 获取特征图信息
    # Ts 8*4*4*4
    # make depth volume
    depths = depths.view([1, 1, dd, 1, 1])
    # 对其进行扩张，扩张到和fmaps维度基本相同
    depths = depths.repeat([batch, num, 1, ht, wd])
    # 根据梯度选取张数
    ii, jj = torch.meshgrid(torch.arange(1), torch.arange(0, num))
    ii = ii.view([-1]).cuda()
    jj = jj.view([-1]).cuda()
    Tii = my_gather(Ts, ii, 1)
    Tjj = my_gather(Ts, jj, 1)
    # 计算对应矩阵 
    Tij = Tjj * TS_inverse(Tii)
    #print(Tij.shape)
    # 将所有深度点，映射到二维空间中
    coords = get_cood(depths, intrinsics, Tij)
    
    volume = my_bilinear_sampler(fmaps, coords)
    
    # 8*128*32*30*40
    volume = volume.view([batch, c*num, dd, ht, wd])
    return volume

