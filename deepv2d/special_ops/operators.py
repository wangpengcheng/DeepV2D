import torch
import os.path as osp


from utils.bilinear_sampler import *

#cholesky_solve = cholesky.solve
def my_gather(input, indexs,dim=1):
    return input.index_select(dim, indexs)


def adj_to_inds(num=-1, adj_list=None):
    """ Convert adjency list into list of edge indicies (ii, jj) = (from, to)"""
    if adj_list is None:
        ii, jj = torch.meshgrid(torch.arange(1), torch.arange(0, num)) # 进行平滑操作
    else:
        n, m = torch.unbind(torch.Tensor(list(adj_list.shape), num=2))
        ii,jj = torch.split(adj_list, [1, m-1], dim=-1)
        ii = ii.repeat(1, m-1)
    ii = torch.reshape(ii, [-1])
    jj = torch.reshape(jj, [-1])
    
    return ii, jj


def backproject_avg(Ts, depths, intrinsics, fmaps, adj_list=None):
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
    # 获取通道数目
    dim = fmaps.shape[2]
    # 获取深度数量
    dd = depths.shape[0]
    # 将特征图进行矩阵分解，获取batch、num、ht和wd等，
    batch, num, _, ht, wd = fmaps.shape[0:] # 获取特征图信息

    # make depth volume
    depths = torch.reshape(depths, [1, 1, dd, 1, 1])
    # 对其进行扩张，扩张到和fmaps维度基本相同
    depths = depths.repeat([batch, 1, 1, ht, wd])
    # 根据梯度选取张数
    ii, jj = adj_to_inds(num, adj_list)
    Tii = Ts.gather(ii) * Ts.gather(ii).inv() # this is just a set of id trans. 转化为se3矩阵
    Tij = Ts.gather(jj) * Ts.gather(ii).inv() # relative camera poses in graph 图形中的相对相机姿势
    # 获取总数量
    num = ii.shape[0]
    # 重新进行维度扩展
    depths = depths.repeat((1, num, 1, 1, 1))

    coords1 = Tii.transform(depths, intrinsics) # 进行坐标转换，转换为x,y,z的三维空间坐标点
    coords2 = Tij.transform(depths, intrinsics) # 坐标2
    # 1*4*30*40*32
    fmap1 = my_gather(fmaps, ii, dim=1).permute(0,1,3,4,2) # 获取数组切片，主要是获取ii中的数据
    fmap2 = my_gather(fmaps, jj, dim=1).permute(0,1,3,4,2) #
    # 将坐标进行维度转换，，将x,y,z->z,x,y 1*4*30*40*32*2
    coords1 = coords1.permute(0, 1, 3, 4, 2, 5)
    coords2 = coords2.permute(0, 1, 3, 4, 2, 5)
    # 对应fmap,的坐标点上，进行双阶线性采样；获取比较准确的坐标样本，作为图像特征值
    fvol1 = bilinear_sampler(fmap1, coords1, batch_dims=2) #1*4*30*40*32*32
    # 双阶段线性采样
    fvol2 = bilinear_sampler(fmap2, coords2, batch_dims=2)
    # 计算特征值  1 4 30 40 32 
    volume = torch.cat([fvol1, fvol2], dim=-1).permute(0, 1, 4, 2, 3)

    if adj_list is None:
        # 将特征值进行重组，相当于特征混合
        volume = torch.reshape(volume, [batch, num, 2*dim, dd,  ht, wd  ])
    else:
        n, m = torch.unbind(torch.Tensor(list(adj_list.shape), num=2))
        volume = torch.reshape(volume, [batch*n,m-1, 2*dim,dd, ht, wd])

    return volume # 3D特征组合


# def backproject_cat(Ts, depths, intrinsics, fmaps):
#     dim = fmaps.get_shape().as_list()[-1]
#     dd = depths.get_shape().as_list()[0]
#     batch, num, ht, wd, _ = torch.unbind(tf.shape(fmaps), num=5)

#     # make depth volume
#     depths = torch.reshape(depths, [1, 1, dd, 1, 1])
#     depths = tf.tile(depths, [batch, num, 1, ht, wd])

#     ii, jj = tf.meshgrid(tf.range(1), tf.range(0, num))
#     ii = torch.reshape(ii, [-1])
#     jj = torch.reshape(jj, [-1])

#     # compute backprojected coordinates
#     Tij = Ts.gather(jj) * Ts.gather(ii).inv()
#     coords = Tij.transform(depths, intrinsics)

#     if use_cuda_backproject:
#         coords = torch.transpose(coords, [0, 3, 4, 2, 1, 5])
#         fmaps = torch.transpose(fmaps, [0, 2, 3, 1, 4])
#         volume = back_project(fmaps, coords)

#     else:
#         coords = torch.transpose(coords, [0, 1, 3, 4, 2, 5])
#         volume = bilinear_sampler(fmaps, coords, batch_dims=2)
#         volume = torch.transpose(volume, [0, 2, 3, 4, 1, 5])

#     volume = torch.reshape(volume, [batch, ht, wd, dd, dim*num])
#     return volume

