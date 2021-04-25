import torch
import numpy as np
import torch.nn.functional as F
import itertools
def th_gather_nd1(x, indices):
    # 获取新文件
    newshape = indices.shape[:-1]+ x.shape[indices.shape[-1]:]
    indices = indices.view(-1, x.shape[-1]).tolist()
    out = torch.cat([x.__getitem__(tuple(i)) for i in indices])
    return x.reshape(newshape)



def my_gather_nd(params, indices):
    '''
    4D example
    params: tensor shaped [n_1, n_2, n_3, n_4] --> 4 dimensional
    indices: tensor shaped [m_1, m_2, m_3, m_4, 4] --> multidimensional list of 4D indices
    
    returns: tensor shaped [m_1, m_2, m_3, m_4]
    
    ND_example
    params: tensor shaped [n_1, ..., n_p] --> d-dimensional tensor
    indices: tensor shaped [m_1, ..., m_i, d] --> multidimensional list of d-dimensional indices
    
    returns: tensor shaped [m_1, ..., m_1]
    '''
    # 输出形状
    out_shape = indices.shape[:-1]
    # 进行降低维度操作
    indices = indices.unsqueeze(0).transpose(0, -1) # roll last axis to fring
    # 
    ndim = indices.shape[0]
    indices = indices.long()
    idx = torch.zeros_like(indices[0], device=indices.device).long()
    m = 1
    
    for i in range(ndim)[::-1]:
        idx += indices[i] * m 
        m *= params.size(i)
    out = torch.take(params, idx)
    return out.view(out_shape)

def torch_gather_nd(params: torch.Tensor,
                    indices: torch.Tensor) -> torch.Tensor:
    """
    Perform the tf.gather_nd on torch.Tensor. Although working, this implementation is
    quite slow and 'ugly'. You should not care to much about performance when using
    this function. I encourage you to think about how to optimize this.

    This function has not been tested properly. It has only been tested empirically
    and shown to yield identical results compared to tf.gather_nd. Still, use at your
    own risk.

    Does not support the `batch_dims` argument that tf.gather_nd does support. This is
    something for future work.

    :param params: (Tensor) - the source Tensor
    :param indices: (LongTensor) - the indices of elements to gather
    :return output: (Tensor) – the destination tensor
    """
    assert indices.dtype == torch.int64, f"indices must be torch.LongTensor, got {indices.dtype}"
    assert indices.shape[-1] <= len(params.shape), f'The last dimension of indices can be at most the rank ' \
                                                   f'of params ({len(params.shape)})'

    # Define the output shape. According to the  documentation of tf.gather_nd, this is:
    # "indices.shape[:-1] + params.shape[indices.shape[-1]:]"
    output_shape = indices.shape[:-1] + params.shape[indices.shape[-1]:]

    # Initialize the output Tensor as an empty one.
    output = torch.zeros(size=output_shape, device=params.device, dtype=params.dtype)

    # indices_to_fill is a list of tuple containing the indices to fill in `output`
    indices_to_fill = list(itertools.product(*[range(x) for x in output_shape[:-1]]))

    # Loop over the indices_to_fill and fill the `output` Tensor
    for idx in indices_to_fill:
        index_value = indices[idx]
        if len(index_value.shape) == 0:
            index_value = torch.Tensor([0, index_value.item()])

        value = params[index_value.view(-1, 1).tolist()].view(-1)
        output[idx] = value

    return output

def gather_nd(image, indicies, batch_dims=0):
    indicies_shape = list(indicies.shape)
    batch_inds = [torch.arange(indicies_shape[i]) for i in range(batch_dims)]
    batch_inds = torch.meshgrid(*batch_inds)
    batch_inds = torch.stack(batch_inds, dim=-1)

    batch_shape, batch_tile = [], []
    for i in range(len(indicies_shape)-1):
        if i < batch_dims:
            batch_shape.append(indicies_shape[i])
            batch_tile.append(1)
        else:
            batch_shape.append(1)
            batch_tile.append(indicies_shape[i])

    batch_shape.append(batch_dims)
    batch_tile.append(1)
    batch_inds = torch.reshape(batch_inds, batch_shape) #1 4 1 1 1 2
    batch_inds = batch_inds.repeat(batch_tile) # 1 4 30 40 32 2
    # 1 4 30 40 32 4 
    indicies = torch.cat([batch_inds, indicies], dim=-1)
    
    return torch_gather_nd(image, indicies)


def my_bilinear_sampler(image, coords):
    """ 
    performs bilinear sampling using coords grid 
    网格坐标双阶线性采样
    主要是对现有的图像，进行双节线性采样
    """
    batch, num, c , ht, wd = image.shape[:]
    # 进行双阶段线性采样
    fmaps = image.view(batch*num, c, 1, ht, wd)
    fmaps = fmaps.repeat(1, 1, num, 1, 1)
    coords = coords.view(batch*num, c, ht, wd, 2)
    
    # 进行坐标分解
    coords_x, coords_y = torch.unbind(coords, dim=-1)
    coords_x = torch.clamp(coords_x, 0, wd-1)
    coords_y = torch.clamp(coords_y, 0, ht-1)
    # 构造num维度
    batch_inds = torch.arange(num, device=torch.device('cuda:0'))
    batch_inds = batch_inds.view([num, 1, 1, 1])
    batch_inds = batch_inds.repeat(batch, c, ht, wd)
    # 进行维度合并,主要是为了适配grid_sample函数
    coords_x = 2.*coords_x/(wd-1) - 1
    coords_y = 2.*coords_y/(ht-1) - 1
    batch_inds = 2.*batch_inds/(num-1) - 1
    # 进行合并
    my_coords = torch.stack([coords_x, coords_y, batch_inds], dim=-1)
    # 
    volmap = F.grid_sample(fmaps, my_coords, mode='bilinear', align_corners=True)
    return volmap

def bilinear_sampler(image, coords, batch_dims=1, return_valid=False):
    """ 
    performs bilinear sampling using coords grid 
    网格坐标双阶线性采样
    主要是对现有的图像，进行双节线性采样
    """
    img_shape = image.shape
    # 获取相对坐标，x和y
    coords_x, coords_y = torch.split(coords, [1, 1], dim=-1)
    # 进行求整数值
    x0 = torch.floor(coords_x)
    # 求取整数值
    y0 = torch.floor(coords_y)
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
    x0 = x0.int()
    x1 = x1.int()
    y0 = y0.int()
    y1 = y1.int()
    # 对数据进行筛选，x对应宽度，避免超过
    x0c = torch.clamp(x0, 0, img_shape[-2]-1)
    x1c = torch.clamp(x1, 0, img_shape[-2]-1)
    # 对数据进行筛选，y对应高度，
    y0c = torch.clamp(y0, 0, img_shape[-3]-1)
    y1c = torch.clamp(y1, 0, img_shape[-3]-1)
    # 
    valid = torch.eq(x0c, x0) & torch.eq(x1c, x1) & \
        torch.eq(y0c, y0) & torch.eq(y1c, y1)
    valid = valid.float()
    # 合成新坐标，主要是个坐标
    coords00 = torch.cat([y0c,x0c], dim=-1)
    coords01 = torch.cat([y0c,x1c], dim=-1)
    coords10 = torch.cat([y1c,x0c], dim=-1)
    coords11 = torch.cat([y1c,x1c], dim=-1)
    # 获取坐标对应的值，对应的值
    img00 = gather_nd(image, coords00, batch_dims=batch_dims)
    img01 = gather_nd(image, coords01, batch_dims=batch_dims)
    img10 = gather_nd(image, coords10, batch_dims=batch_dims)
    img11 = gather_nd(image, coords11, batch_dims=batch_dims)
    # 进行权重更新，计算总的值；
    out = w00*img00 + w01*img01 + w10*img10 + w11*img11
    if return_valid:
        return valid*out, valid
    
    return valid * out