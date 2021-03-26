import tensorflow as tf
import numpy as np

def gather_nd(image, indicies, batch_dims=0):
    # 获取坐标形状
    indicies_shape = tf.shape(indicies) # 获取目标参数形状
    # 获取前两个维度，batch,n
    batch_inds = [tf.range(indicies_shape[i]) for i in range(batch_dims)] # 1*4
    # 进行划分
    batch_inds = tf.meshgrid(*batch_inds, indexing='ij') # 将其转换为矩阵
    # 将其组合
    batch_inds = tf.stack(batch_inds, axis=-1) # 1 4 2
    # 步长形状,步长重合 ，构造新的形状，主要是方便维度转换
    batch_shape, batch_tile = [], [] 
    for i in range(len(indicies.get_shape().as_list())-1):
        # 进行原来的维度
        if i < batch_dims: # 
            batch_shape.append(indicies_shape[i])
            batch_tile.append(1)
        else:
            batch_shape.append(1)
            batch_tile.append(indicies_shape[i])
    # 
    batch_shape.append(batch_dims) # 1 4 1 1 1 2
    batch_tile.append(1) # 1 1 30 40 32 1 
    # 步长增长
    batch_inds = tf.reshape(batch_inds, batch_shape) #1 4 1 1 1 2
    # 将所有元素进行复制
    batch_inds = tf.tile(batch_inds, batch_tile)  # 1 4 30 40 32 2
    # 1,4,30,40,32,4  1,4,30,40,32
    indicies = tf.concat([batch_inds, indicies], axis=-1)
    return tf.gather_nd(image, indicies)

def bilinear_sampler(image, coords, batch_dims=1, return_valid=False):
    """ 
    performs bilinear sampling using coords grid 
    网格坐标双阶线性采样
    主要是对现有的图像，进行双节线性采样
    """
    img_shape = tf.shape(image)
    # 获取相对坐标，x和y
    coords_x, coords_y = tf.split(coords, [1, 1], axis=-1)
    # 进行求整数值
    x0 = tf.floor(coords_x)
    # 求取整数值
    y0 = tf.floor(coords_y)
    # 向上计算
    x1 = x0 + 1.0
    # 向上计算
    y1 = y0 + 1.0
    # 计算各个方向上的梯度
    w00 = (x1-coords_x) * (y1-coords_y)
    w01 = (coords_x-x0) * (y1-coords_y)
    w10 = (x1-coords_x) * (coords_y-y0)
    w11 = (coords_x-x0) * (coords_y-y0)
    # 将坐标转换为int类型
    x0 = tf.cast(x0, 'int32')
    x1 = tf.cast(x1, 'int32')
    y0 = tf.cast(y0, 'int32')
    y1 = tf.cast(y1, 'int32')
    # 对数据进行筛选，x对应宽度，避免超过
    x0c = tf.clip_by_value(x0, 0, img_shape[-2]-1)
    x1c = tf.clip_by_value(x1, 0, img_shape[-2]-1)
    # 对数据进行筛选，y对应高度，
    y0c = tf.clip_by_value(y0, 0, img_shape[-3]-1)
    y1c = tf.clip_by_value(y1, 0, img_shape[-3]-1)
    # 是否相等--没有梯度
    valid = tf.equal(x0c, x0) & tf.equal(x1c, x1) & \
        tf.equal(y0c, y0) & tf.equal(y1c, y1)
    valid = tf.cast(valid, 'float32')
    # 合成新坐标，主要是个坐标，[x,y] [x+1,y] [x,y+1] [x+1,y+1]
    coords00 = tf.concat([y0c,x0c], axis=-1)
    coords01 = tf.concat([y0c,x1c], axis=-1)
    coords10 = tf.concat([y1c,x0c], axis=-1)
    coords11 = tf.concat([y1c,x1c], axis=-1)
    # 获取坐标对应的值，对应的值
    img00 = gather_nd(image, coords00, batch_dims=batch_dims) #1 4 30 42 32 32
    img01 = gather_nd(image, coords01, batch_dims=batch_dims)
    img10 = gather_nd(image, coords10, batch_dims=batch_dims)
    img11 = gather_nd(image, coords11, batch_dims=batch_dims)
    # 进行权重更新，计算总的值；
    out = w00*img00 + w01*img01 + w10*img10 + w11*img11
    if return_valid:
        return valid*out, valid
    
    return valid * out