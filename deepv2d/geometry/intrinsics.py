import tensorflow as tf
import numpy as np
from utils.einsum import einsum
# 将相机参数转换为矩阵
def intrinsics_vec_to_matrix(kvec):
    fx, fy, cx, cy = tf.unstack(kvec, num=4, axis=-1)
    z = tf.zeros_like(fx) #零阶矩阵
    o = tf.ones_like(fx) #1阶矩阵

    K = tf.stack([fx, z, cx, z, fy, cy, z, z, o], axis=-1)
    K = tf.reshape(K, kvec.get_shape().as_list()[:-1] + [3,3]) # 相机矩阵拼接
    return K

def intrinsics_matrix_to_vec(kmat):
    fx = kmat[..., 0, 0]
    fy = kmat[..., 1, 1]
    cx = kmat[..., 0, 2]
    cy = kmat[..., 1, 2]
    return tf.stack([fx, fy, cx, cy], axis=-1) # 提取所有帧的相机内参，并转化为一维度数据

def update_intrinsics(intrinsics, delta_focal):
    # 将位姿信息转转为kvect
    kvec = intrinsics_matrix_to_vec(intrinsics)
    fx, fy, cx, cy = tf.unstack(kvec, num=4, axis=-1)
    df = tf.squeeze(delta_focal, -1)

    # update the focal lengths
    fx = tf.exp(df) * fx
    fy = tf.exp(df) * fy

    kvec = tf.stack([fx, fy, cx, cy], axis=-1)
    kmat = intrinsics_vec_to_matrix(kvec)
    return kmat
# 对深度进行重新缩放
def rescale_depth(depth, downscale=4):
    depth = tf.expand_dims(depth, axis=-1) # 将所有维度扩展
    new_shape = tf.shape(input=depth)[1:3] // downscale
    depth = tf.image.resize(depth, new_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) # 进行插值和
    return tf.squeeze(depth, axis=-1)

def rescale_depth_and_intrinsics(depth, intrinsics, downscale=4):
    sc = tf.constant([1.0/downscale, 1.0/downscale, 1.0], dtype=tf.float32) # 构建缩放矩阵
    intrinsics = einsum('...ij,i->...ij', intrinsics, sc) # 求点乘将值缩放为原来的1/4
    depth = rescale_depth(depth, downscale=downscale) #
    return depth, intrinsics

def rescale_depths_and_intrinsics(depth, intrinsics, downscale=4):
    batch, frames, height, width = [tf.shape(input=depth)[i] for i in range(4)] # 获取数据维度信息
    depth = tf.reshape(depth, [batch*frames, height, width]) # 将深度图，转换为三维的叠加产物，注意这里维度缩减了
    depth, intrinsics = rescale_depth_and_intrinsics(depth, intrinsics, downscale)
    depth = tf.reshape(depth,
        tf.concat(([batch, frames], tf.shape(input=depth)[1:]), axis=0))
    return depth, intrinsics
