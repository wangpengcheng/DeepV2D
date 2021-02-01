import tensorflow as tf
import os.path as osp

from tensorflow.python.framework import ops
from utils.bilinear_sampler import *

filename = osp.join(osp.dirname(__file__), 'backproject.so')
if osp.isfile(filename):
    _backproject_module = tf.load_op_library(filename)

    back_project = _backproject_module.back_project
    back_project_grad = _backproject_module.back_project_grad


    @ops.RegisterGradient("BackProject")
    def _back_project_grad(op, grad):
        inputs = op.inputs[0]
        coords = op.inputs[1]
        inputs_grad, coords_grad = back_project_grad(inputs, coords, grad)

        return [inputs_grad, coords_grad]

    use_cuda_backproject = True

else:
    print("Could not import cuda Backproject Module, using python implementation")
    use_cuda_backproject = False


def adj_to_inds(num=-1, adj_list=None):
    """ Convert adjency list into list of edge indicies (ii, jj) = (from, to)
        将邻接列表转换为边缘索引列表
    """
    if adj_list is None:
        ii, jj = tf.meshgrid(tf.range(1), tf.range(1, num)) # 进行平滑操作,ii都是1，jj是1~num,但是两个的大小都相同
    else:
        # 进行矩阵分解，这里是按照frame进行拆解
        n, m = tf.unstack(tf.shape(adj_list), num=2)
        #
        ii, jj = tf.split(adj_list, [1, m-1], axis=-1)
        ii = tf.tile(ii, [1, m-1])

    ii = tf.reshape(ii, [-1])
    jj = tf.reshape(jj, [-1])
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
    dim = fmaps.get_shape().as_list()[-1]
    # 获取深度数量
    dd = depths.get_shape().as_list()[0]
    # 将特征图进行矩阵分解，获取batch、num、ht和wd等，
    batch, num, ht, wd, _ = tf.unstack(tf.shape(fmaps), num=5) # 获取特征图信息

    # make depth volume
    depths = tf.reshape(depths, [1, 1, dd, 1, 1])
    # 对其进行扩张，扩张到和fmaps维度基本相同
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

    fmap1 = tf.gather(fmaps, ii, axis=1) # 获取数组切片，主要是获取ii中的数据
    fmap2 = tf.gather(fmaps, jj, axis=1) #
    # 使用cuda反向映射
    if use_cuda_backproject:
        coords = tf.stack([coords1, coords2], axis=-2)
        coords = tf.reshape(coords, [batch*num, dd, ht, wd, 2, 2])
        coords = tf.transpose(coords, [0, 2, 3, 1, 4, 5])

        fmap1 = tf.reshape(fmap1, [batch*num, ht, wd, dim]) # 调整特征图
        fmap2 = tf.reshape(fmap2, [batch*num, ht, wd, dim])
        fmaps_stack = tf.stack([fmap1, fmap2], axis=-2)

        # cuda backprojection operator
        volume = back_project(fmaps_stack, coords)

    else:
        # 将坐标进行维度转换，，将x,y,z->z,x,y
        coords1 = tf.transpose(coords1, [0, 1, 3, 4, 2, 5])
        coords2 = tf.transpose(coords2, [0, 1, 3, 4, 2, 5])
        # 对应fmap,的坐标点上，进行双阶线性采样；获取比较准确的坐标样本，作为图像特征值
        fvol1 = bilinear_sampler(fmap1, coords1, batch_dims=2)
        fvol2 = bilinear_sampler(fmap2, coords2, batch_dims=2)
        # 计算特征值
        volume = tf.concat([fvol1, fvol2], axis=-1)

    if adj_list is None:
        # 将特征值进行重组，相当于特征混合
        volume = tf.reshape(volume, [batch, num, ht, wd, dd, 2*dim])
    else:
        n, m = tf.unstack(tf.shape(adj_list), num=2)
        volume = tf.reshape(volume, [batch*n, m-1, ht, wd, dd, 2*dim])

    return volume # 3D特征组合


def backproject_cat(Ts, depths, intrinsics, fmaps):
    dim = fmaps.get_shape().as_list()[-1]
    dd = depths.get_shape().as_list()[0]
    batch, num, ht, wd, _ = tf.unstack(tf.shape(fmaps), num=5)

    # make depth volume
    depths = tf.reshape(depths, [1, 1, dd, 1, 1])
    depths = tf.tile(depths, [batch, num, 1, ht, wd])

    ii, jj = tf.meshgrid(tf.range(1), tf.range(0, num))
    ii = tf.reshape(ii, [-1])
    jj = tf.reshape(jj, [-1])

    # compute backprojected coordinates
    Tij = Ts.gather(jj) * Ts.gather(ii).inv()
    coords = Tij.transform(depths, intrinsics)

    if use_cuda_backproject:
        coords = tf.transpose(coords, [0, 3, 4, 2, 1, 5])
        fmaps = tf.transpose(fmaps, [0, 2, 3, 1, 4])
        volume = back_project(fmaps, coords)

    else:
        coords = tf.transpose(coords, [0, 1, 3, 4, 2, 5])
        volume = bilinear_sampler(fmaps, coords, batch_dims=2)
        volume = tf.transpose(volume, [0, 2, 3, 4, 1, 5])

    volume = tf.reshape(volume, [batch, ht, wd, dd, dim*num])
    return volume


###
### Operators for manipulating gradients during backward pass
###

from tensorflow.python.framework import function


def clip_dangerous_gradients_grad(op, grad):
    grad = tf.where(tf.is_nan(grad),    tf.zeros_like(grad), grad)
    grad = tf.where(tf.abs(grad)>1e-3, tf.zeros_like(grad), grad)
    return [grad]

@function.Defun(tf.float32,
				python_grad_func=clip_dangerous_gradients_grad,
				shape_func=lambda op: [op.inputs[0].shape])
def clip_dangerous_gradients(x):
	return x

def clip_nan_gradients_grad(op, grad):
    grad = tf.where(tf.is_nan(grad),    tf.zeros_like(grad), grad)
    return [grad]

@function.Defun(tf.float32,
				python_grad_func=clip_nan_gradients_grad,
				shape_func=lambda op: [op.inputs[0].shape])
def clip_nan_gradients(x):
	return x