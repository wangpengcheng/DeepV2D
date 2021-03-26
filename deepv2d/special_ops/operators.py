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
<<<<<<< HEAD
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
    re = torch.bmm(Tij, PT)
    re = re.view(batch, num, c, ht, wd, 4)


    X, Y, Z, one = torch.unbind(re, dim=-1)
    pt1 = torch.stack([X, Y, Z], dim = -1)
    coords = project(pt1, intrinsics)

    return coords 

def backproject_avg(
            Ts, 
            depths, 
            intrinsics, 
            fmaps, 
            back_project
            ):
=======
        ii, jj = tf.meshgrid(tf.range(1), tf.range(1, num)) # 进行平滑操作,ii都是1，jj是1~num,但是两个的大小都相同
    else:
        # 进行矩阵分解，这里是按照frame进行拆解
        n, m = tf.unstack(tf.shape(adj_list), num=2)
        #
        ii, jj = tf.split(adj_list, [1, m-1], axis=-1)
        ii = tf.tile(ii, [1, m-1])
    # 将其转换为1维度坐标
    ii = tf.reshape(ii, [-1])
    jj = tf.reshape(jj, [-1])
    return ii, jj


def backproject_avg(Ts, depths, intrinsics, fmaps, adj_list=None):
>>>>>>> test_run
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
<<<<<<< HEAD
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
    depths = torch.reshape(depths, [1, 1, dd, 1, 1])
    # 对其进行扩张，扩张到和fmaps维度基本相同
    depths = depths.repeat([batch, num, 1, ht, wd])
    # 根据梯度选取张数
    ii, jj = torch.meshgrid(torch.arange(1), torch.arange(0, num))
    ii = ii.view([-1]).tolist()
    jj = jj.view([-1]).tolist()
    Tii = Ts[:,ii,]
    Tjj = Ts[:,jj,]
    # 计算对应矩阵 
    Tij = Tjj * torch.inverse(Tii)
    # 将所有深度点，映射到三维空间中
    coords = get_cood(depths, intrinsics, Tij)
    
    volume = my_bilinear_sampler(fmaps, coords)
    
    # 8*128*32*30*40
    volume = torch.reshape(volume, [batch, c*num, dd, ht, wd])
    return volume
=======
    # 获取通道数目
    dim = fmaps.get_shape().as_list()[-1]
    # 获取深度数量
    dd = depths.get_shape().as_list()[0]
    # 将特征图进行矩阵分解，获取batch、num、ht和wd等，
    batch, num, ht, wd, _ = tf.unstack(tf.shape(fmaps), num=5) # 获取特征图信息

    # make depth volume
    # 创建深度图，注意这里的中间值是维度，表示深度的映射值
    depths = tf.reshape(depths, [1, 1, dd, 1, 1])
    # 对其进行扩张，扩张到和fmaps维度基本相同，保证每个batch和frame对应一张深度图
    depths = tf.tile(depths, [batch, 1, 1, ht, wd])
    # 根据梯度选取张数
    ii, jj = adj_to_inds(num, adj_list)
    Tii = Ts.gather(ii) * Ts.gather(ii).inv() # this is just a set of id trans. 转化为se3矩阵
    Tij = Ts.gather(jj) * Ts.gather(ii).inv() # relative camera poses in graph 图形中的相对相机姿势
    # 获取总数量
    num = tf.shape(ii)[0]
    # 重新进行维度扩展
    depths = tf.tile(depths, [1, num, 1, 1, 1])

    coords1 = Tii.transform(depths, intrinsics) # 进行坐标转换，转换为x,y,z的三维空间坐标点
    coords2 = Tij.transform(depths, intrinsics) # 坐标2

    fmap1 = tf.gather(fmaps, ii, axis=1) # 获取数组切片，主要是获取ii中的数据 1.4,30,40,32
    fmap2 = tf.gather(fmaps, jj, axis=1) #
    # 使用cuda反向映射
    if use_cuda_backproject:
        coords = tf.stack([coords1, coords2], axis=-2) #1 4 32 30 40 2 2
        coords = tf.reshape(coords, [batch*num, dd, ht, wd, 2, 2]) # 4 32 30 40 2 2
        coords = tf.transpose(coords, [0, 2, 3, 1, 4, 5]) # 4 30 40 32 2 2

        fmap1 = tf.reshape(fmap1, [batch*num, ht, wd, dim]) # 调整特征图 4 30 40 32
        fmap2 = tf.reshape(fmap2, [batch*num, ht, wd, dim]) # 4 30 40 32
        fmaps_stack = tf.stack([fmap1, fmap2], axis=-2) # 4 30 40 2 32

        # cuda backprojection operator
        volume = back_project(fmaps_stack, coords) # 1 4 30 40 32 64

    else:
        # 将坐标进行维度转换，，将x,y,z->z,x,y
        coords1 = tf.transpose(coords1, [0, 1, 3, 4, 2, 5]) # 1 4 30 40 32 2
        coords2 = tf.transpose(coords2, [0, 1, 3, 4, 2, 5])
        # 对应fmap,的坐标点上，进行双阶线性采样；获取比较准确的坐标样本，作为图像特征值
        fvol1 = bilinear_sampler(fmap1, coords1, batch_dims=2) # 1 4 30 40 32 32
        # 双阶段线性采样
        fvol2 = bilinear_sampler(fmap2, coords2, batch_dims=2)
        # 计算特征值
        volume = tf.concat([fvol1, fvol2], axis=-1) # 1 4 30 40 32 64

    if adj_list is None:
        # 将特征值进行重组，相当于特征混合
        volume = tf.reshape(volume, [batch, num, ht, wd, dd, 2*dim]) # 1 4 30 40 32 64
    else:
        n, m = tf.unstack(tf.shape(adj_list), num=2)
        volume = tf.reshape(volume, [batch*n, m-1, ht, wd, dd, 2*dim])

    return volume # 3D特征组合
>>>>>>> test_run


def backproject_cat(Ts, depths, intrinsics, fmaps):
    # 获取通道数目
<<<<<<< HEAD
    dim = fmaps.shape[2]
    # 获取深度数量
    dd = depths.shape[0]
    batch, num, ht, wd, _ = torch.unbind(tf.shape(fmaps), num=5)

    # make depth volume
    depths = torch.reshape(depths, [1, 1, dd, 1, 1])
=======
    dim = fmaps.get_shape().as_list()[-1]
    # 获取深度数量
    dd = depths.get_shape().as_list()[0]
    # # 将特征图进行矩阵分解，获取batch、num、ht和wd等，
    batch, num, ht, wd, _ = tf.unstack(tf.shape(fmaps), num=5)

    # make depth volume
    depths = tf.reshape(depths, [1, 1, dd, 1, 1])
    # 将其进行扩张到和fmaps维度相同，注意这里的num进行了保留
>>>>>>> test_run
    depths = tf.tile(depths, [batch, num, 1, ht, wd])
    # 进行平滑操作
    ii, jj = tf.meshgrid(tf.range(1), tf.range(0, num))
    ii = torch.reshape(ii, [-1])
    jj = torch.reshape(jj, [-1])

    # compute backprojected coordinates
    Tij = Ts.gather(jj) * Ts.gather(ii).inv()
    coords = Tij.transform(depths, intrinsics)

    if use_cuda_backproject:
<<<<<<< HEAD
        coords = torch.transpose(coords, [0, 3, 4, 2, 1, 5])
        fmaps = torch.transpose(fmaps, [0, 2, 3, 1, 4])
=======
        coords = tf.transpose(coords, [0, 3, 4, 2, 1, 5])
        fmaps = tf.transpose(fmaps, [0, 2, 3, 1, 4])
        # 反向计算
>>>>>>> test_run
        volume = back_project(fmaps, coords)

    else:
        coords = torch.transpose(coords, [0, 1, 3, 4, 2, 5])
        volume = bilinear_sampler(fmaps, coords, batch_dims=2)
        volume = torch.transpose(volume, [0, 2, 3, 4, 1, 5])

    volume = torch.reshape(volume, [batch, ht, wd, dd, dim*num])
    return volume

