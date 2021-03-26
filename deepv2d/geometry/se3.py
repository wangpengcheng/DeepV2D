import numpy as np
import torch 
from utils.einsum import *
import os
"""
SO3 and SE3 operations, exponentials and logarithms adapted from Sophus
"""


MIN_THETA = 1e-4

def matdotv(A,b):
    """
<<<<<<< HEAD
    通用矩阵乘法
=======
    实现矩阵乘法
>>>>>>> test_run
    Args:
        A ([type]): [description]
        b ([type]): [description]

    Returns:
        [type]: [description]
    """
<<<<<<< HEAD
    return torch.squeeze(torch.matmul(A, torch.unsqueeze(b, -1)), -1)
=======
    return tf.squeeze(tf.matmul(A, tf.expand_dims(b, -1)), -1)
>>>>>>> test_run

def hat(a):
    """
    将向量转换为反对称矩阵，方便进行外积运算
    Args:
        a ([type]): [description]

    Returns:
        [type]: [description]
    """
<<<<<<< HEAD
    a1, a2, a3 = torch.split(a, [1,1,1], dim=-1)
    zz = torch.zeros_like(a1)

    ax = torch.stack([
        torch.cat([zz, -a3, a2], dim=-1),
        torch.cat([a3, zz, -a1], dim=-1),
        torch.cat([-a2, a1, zz], dim=-1)
    ], dim=-2)
=======
    # 获取基础坐标系下的值
    a1, a2, a3 = tf.split(a, [1,1,1], axis=-1)
    # 创建同维度的同阶矩阵
    zz = tf.zeros_like(a1)
    # 进行维度合并
    ax = tf.stack([
        tf.concat([zz,-a3,a2], axis=-1),
        tf.concat([a3,zz,-a1], axis=-1),
        tf.concat([-a2,a1,zz], axis=-1)
    ], axis=-2)
>>>>>>> test_run

    return ax
    

### quaternion functions ###

def quaternion_rotate_point(q, pt, eq=None):
    """
<<<<<<< HEAD
    四元数点旋转
    Args:
        q ([type]): 四元数点
        pt ([type]): 旋转点
=======
    四元数旋转点
    Args:
        q ([type]): [description]
        pt ([type]): [description]
>>>>>>> test_run
        eq ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    if eq is None:
<<<<<<< HEAD
        # 获取，实部和虚部
        w, vec = torch.split(q, [1, 3], dim=-1)
        uv = 2*matdotv(hat(vec), pt) # 2pq^
=======
        # 分离原始的三维向量和，W权重指标
        w, vec = tf.split(q, [1, 3], axis=-1)
        # 计算旋转变化
        uv = 2*matdotv(hat(vec), pt)
        # 计算新的点结果
>>>>>>> test_run
        return pt + w*uv + matdotv(hat(vec), uv)
    else:
        w, vec = torch.split(q, [1, 3], dim=-1)
        uv1 = 2*einsum(eq, hat(w*vec), pt)
        uv2 = 2*einsum(eq, hat(vec), pt)
        return pt + uv1 + einsum(eq, hat(vec), uv2)

def quaternion_rotate_matrix(q, mat, eq=None):
    if eq is None:
        w, vec = torch.split(q, [1, 3], dim=-1)
        uv = 2*torch.matmul(hat(vec), mat)
        return mat + w*uv + torch.matmul(hat(vec), uv)
    else:
        w, vec = torch.split(q, [1, 3], dim=-1)
        uv1 = 2*einsum(eq, hat(w*vec), mat)
        uv2 = 2*einsum(eq, hat(vec), mat)
        return mat + uv1 + einsum(eq, hat(vec), uv2)

def quaternion_inverse(q):
    return q * [1, -1, -1, -1]

def quaternion_multiply(a, b):
    """
    两个四元组的乘法
    Args:
        a ([type]): [description]
        b ([type]): [description]

    Returns:
        [type]: [description]
    """
    aw, ax, ay, az = torch.split(a, [1, 1, 1, 1], dim=-1)
    bw, bx, by, bz = torch.split(b, [1, 1, 1, 1], dim=-1)
    
    q = torch.cat([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by + ay * bw + az * bx - ax * bz,
        aw * bz + az * bw + ax * by - ay * bx,
    ], dim=-1)

    return q

def quaternion_to_matrix(q):
    # 四元组转换为旋转矩阵
    w, x, y, z = torch.split(q, [1,1,1,1], dim=-1)

    r11 = 1 - 2 * y**2 - 2 * z**2
    r12 = 2 * x * y - 2 * w * z
    r13 = 2 * z * x + 2 * w * y

    r21 = 2 * x * y + 2 * w * z
    r22 = 1 - 2 * x**2 - 2 * z**2
    r23 = 2 * y * z - 2 * w * x

    r31 = 2 * z * x - 2 * w * y
    r32 = 2 * y * z + 2 * w * x
    r33 = 1 - 2 * x**2 - 2 * y**2
    
    R = torch.stack([
        torch.cat([r11,r12,r13], dim=-1),
        torch.cat([r21,r22,r23], dim=-1),
        torch.cat([r31,r32,r33], dim=-1)
    ], dim=-2)

    return R

def rotation_matrix_to_quaternion(R):
    """
    旋转矩阵转换为四元组
    Args:
        R ([type]): 旋转矩阵

    Returns:
        [type]: [description]
    """
    Rxx, Ryx, Rzx = R[...,0,0], R[...,0,1], R[...,0,2]
    Rxy, Ryy, Rzy = R[...,1,0], R[...,1,1], R[...,1,2]
    Rxz, Ryz, Rzz = R[...,2,0], R[...,2,1], R[...,2,2]

    zz = torch.zeros_like(Rxx)
    k1 = torch.stack([Rxx-Ryy-Rzz, zz, zz, zz], dim=-1)
    k2 = torch.stack([Ryx+Rxy, Ryy-Rxx-Rzz, zz, zz], dim=-1)
    k3 = torch.stack([Rzx+Rxz, Rzy+Ryz, Rzz-Rxx-Ryy,zz], dim=-1)
    k4 = torch.stack([Ryz-Rzy, Rzx-Rxz, Rxy-Ryx, Rxx+Ryy+Rzz], dim=-1)

    K = torch.stack([k1, k2, k3, k4], dim=-2)
    eigvals, eigvecs = np.eigh(K) # 共轭矩阵特征分解

    x, y, z, w = torch.split(eigvecs[...,-1], [1,1,1,1], dim=-1)
    qvec = torch.cat([w, x, y, z], dim=-1)
    qvec /=  torch.sqrt(torch.sum(qvec**2, dim=-1, keepdims=True))

    return qvec * torch.sign(w)

def so3_expm_and_theta(omega):
    """ omega in \so3 """
    theta_sq = torch.sum(omega**2, dim=-1)
    theta = torch.sqrt(theta_sq)
    half_theta = 0.5*theta

    ### small ###
    imag_factor = torch.nonzero(theta>MIN_THETA, 
        torch.sin(half_theta) / (theta + 1e-12), 
        0.5 - (1.0/48.0)*theta_sq + (1.0/3840.0)*theta_sq*theta_sq)

    real_factor = torch.nonzero(theta>MIN_THETA, torch.cos(half_theta),
        1.0 - (1.0/8.0)*theta_sq + (1.0/384.0)*theta_sq*theta_sq)

    qw = real_factor
    qx = imag_factor * omega[...,0]
    qy = imag_factor * omega[...,1]
    qz = imag_factor * omega[...,2]

    quat = torch.stack([qw, qx, qy, qz], dim=-1)
    return quat, theta
        
def so3_logm_and_theta(so3):
    w, vec = torch.split(so3, [1,3], dim=-1)
    squared_n = torch.sum(vec**2, dim=-1, keepdims=True)
    n = torch.sqrt(squared_n)

    two_atan_nbyw_by_n = torch.nonzero(n<MIN_THETA,
        2/w - w*squared_n / (w*w*w),
        2*torch.atan(n/w) / (n+1e-12))

    theta = two_atan_nbyw_by_n * n
    omega = two_atan_nbyw_by_n * vec
    return omega, theta

def se3_expm(xi):
    """ xi in \se3 """
    tau, omega = torch.split(xi, [3, 3], dim=-1)
    q, theta = so3_expm_and_theta(omega)


    theta = theta[...,np.newdim,np.newdim]
    theta = torch.Tensor.repeat(theta, 
        torch.cat([torch.ones_like(q.shape[:-1]), [3,3]], dim=-1))

    theta_sq = theta * theta
    Omega = hat(omega)
    Omega_sq = torch.matmul(Omega, Omega)

    Vs = torch.eye(3, batch_shape=(xi.shape)[:-1]) + \
         (1-torch.cos(theta)) / (theta_sq + 1e-12) * Omega + \
         (theta - torch.sin(theta)) / (theta_sq*theta + 1e-12) * Omega_sq

    V = torch.nonzero(theta<MIN_THETA, quaternion_to_matrix(q), Vs)
    t = matdotv(V, tau)
    return q, t

def se3_logm(so3, t):
    omega, theta = so3_logm_and_theta(so3)
    Omega = hat(omega)
    Omega_sq = torch.matmul(Omega, Omega)

    theta = theta[...,tf.newdim]
    theta = torch.Tensor.repeat(
        theta,
        torch.cat([torch.ones_like(omega.shape[:-1]), [3, 3]], dim=-1)
        )
    half_theta = 0.5*theta

    Vinv_approx = torch.eye(3, batch_shape=(omega.shape)[:-1]) - \
        0.5*Omega + (1.0/12.0) * Omega_sq

    Vinv_exact = torch.eye(3, batch_shape=(omega.shape)[:-1]) - \
        0.5*Omega + (1-theta*torch.cos(half_theta) / \
        (2*torch.sin(half_theta)+1e-12)) / (theta*theta + 1e-12) * Omega_sq

    Vinv = torch.nonzero(theta<MIN_THETA, Vinv_approx, Vinv_exact)
    tau = matdotv(Vinv, t)

    upsilon = torch.cat([tau, omega], dim=-1)
    return upsilon

### matrix functions ###

def se3_matrix_inverse(G):
<<<<<<< HEAD
    """ Invert SE3 matrix """
    inp_shape = G.shape
    G = torch.reshape(G, [-1, 4, 4])

    R, t = G[:, :3, :3], G[:, :3, 3:]
    R = R.permute((0, 2, 1))
    t = -torch.matmul(R, t)

    filler = torch.Tensor([0.0, 0.0, 0.0, 1.0])
    filler = torch.reshape(filler, [1, 1, 4])
    filler = torch.Tensor.repeat(filler, [G.shape[0], 1, 1])

    Ginv = torch.cat([R, t], dim=-1)
    Ginv = torch.cat([Ginv, filler], dim=-2)
    return torch.reshape(Ginv, inp_shape)

=======
    """ Invert SE3 matrix 
    se3维度变换
    """
    # 获取输入数据
    inp_shape = tf.shape(G)
    # 进行变换
    G = tf.reshape(G, [-1, 4, 4])

    R, t = G[:, :3, :3], G[:, :3, 3:]
    R = tf.transpose(R, [0, 2, 1])
    t = -tf.matmul(R, t)
    # 设置静态变量
    filler = tf.constant([0.0, 0.0, 0.0, 1.0])
    # 
    filler = tf.reshape(filler, [1, 1, 4])
    # 扩展到相同维度
    filler = tf.tile(filler, [tf.shape(G)[0], 1, 1])
    # 进行组合
    Ginv = tf.concat([R, t], axis=-1)
    # 进行组合
    Ginv = tf.concat([Ginv, filler], axis=-2)
    return tf.reshape(Ginv, inp_shape)
>>>>>>> test_run

def _se3_matrix_expm_grad(grad):
    """
    梯度计算操作
    Args:
        op ([type]): [description]
        grad ([type]): [description]

    Returns:
        [type]: [description]
    """
    grad_upsilon_omega = torch.stack([
        grad[..., 0, 3],
        grad[..., 1, 3],
        grad[..., 2, 3],
        grad[..., 2, 1] - grad[..., 1, 2],
        grad[..., 0, 2] - grad[..., 2, 0],
        grad[..., 1, 0] - grad[..., 0, 1]
    ], dim=-1)

    return grad_upsilon_omega

def _se3_matrix_expm_shape(op):
    return [op.inputs[0].shape[:-1] + [4, 4]]

<<<<<<< HEAD
def se3_matrix_expm(upsilon_omega):
=======
@function.Defun(tf.float32,
        python_grad_func=_se3_matrix_expm_grad,
        shape_func = _se3_matrix_expm_shape
        )
def se3_matrix_expm(
    upsilon_omega
    ):
>>>>>>> test_run
    """ se3 matrix exponential se(3) -> SE(3), works for arbitrary batch dimensions
    将李代数se(3)转变为SE(3) 特殊欧式群
    主要是进行指数映射关系
    - Note: gradient is overridden with _se3_matrix_expm_grad, which approximates 
    gradient for small upsilon_omega
    - 注意，重载了梯度函数
    """

    eps=1e-12
<<<<<<< HEAD
    inp_shape = upsilon_omega.shape
    out_shape = torch.cat([inp_shape[:-1], [4,4]], dim=-1)

    upsilon_omega = torch.reshape(upsilon_omega, [-1, 6])
    batch = (upsilon_omega.shape)[0]
    v, w = torch.split(upsilon_omega, [3, 3], dim=-1)

    theta_sq = torch.sum(w**2, dim=1)
    theta_sq = torch.reshape(theta_sq, [-1, 1, 1])
=======
    # 获取输入维度
    inp_shape = tf.shape(upsilon_omega)
    # 输出维度n*4*4的矩阵
    out_shape = tf.concat([inp_shape[:-1], [4,4]], axis=-1)
    # 进行重新设置维度
    upsilon_omega = tf.reshape(upsilon_omega, [-1, 6])
    # 批次
    batch = tf.shape(upsilon_omega)[0]
    # 
    v, w = tf.split(upsilon_omega, [3, 3], axis=-1)
    # 计算对应角度值
    theta_sq = tf.reduce_sum(w**2, axis=1)
    theta_sq = tf.reshape(theta_sq, [-1, 1, 1])
>>>>>>> test_run

    theta = torch.sqrt(theta_sq)
    theta_po4 = theta_sq * theta_sq

    wx = hat(w)
    wx_sq = torch.matmul(wx, wx)
    I = torch.eye(3, batch_shape=[batch])

    ### taylor approximations ###
    R1 =  I + (1.0 - (1.0/6.0)*theta_sq + (1.0/120.0)*theta_po4) * wx + \
        (0.5 - (1.0/12.0)*theta_sq + (1.0/720.0)*theta_po4) * wx_sq
    
    V1 = I + (0.5 - (1.0/24.0)*theta_sq + (1.0/720.0)*theta_po4)*wx + \
        ((1.0/6.0) - (1.0/120.0)*theta_sq + (1.0/5040.0)*theta_po4)*wx_sq

    ### exact values ###
    R2 = I + (torch.sin(theta) / (theta+eps)) * wx +\
        ((1 - torch.cos(theta)) / (theta_sq+eps)) * wx_sq

    V2 = I + ((1 - torch.cos(theta)) / (theta_sq + eps)) * wx + \
        ((theta - torch.sin(theta))/(theta_sq*theta + eps)) * wx_sq

    R = torch.nonzero(theta[:, 0, 0]<MIN_THETA, R1, R2)
    V = torch.nonzero(theta[:, 0, 0]<MIN_THETA, V1, V2)

    t = torch.matmul(V, tf.expand_dims(v, -1))
    fill = tf.constant([0, 0, 0, 1], dtype=tf.float32)

    fill = torch.reshape(fill, [1, 1, 4])
    fill = torch.Tensor.repeat(fill, [batch, 1, 1])

    G = torch.cat([R, t], dim=2)
    G = torch.cat([G, fill], dim=1)
    G = torch.reshape(G, out_shape)
    return G

class Se3MatrixExpm(torch.autograd.Function):
    """
    自定义相关结构
    Args:
        torch ([type]): [description]
    """
    def forward(self, input_):
        # 在forward中，需要定义MyReLU这个运算的forward计算过程
        # 同时可以保存任何在后向传播中需要使用的变量值
        self.save_for_backward(input_)         # 将输入保存起来，在backward时使用
        output = se3_matrix_expm(input_) # 向前计算
        return output

    def backward(self, grad_output):
        # 根据BP算法的推导（链式法则），dloss / dx = (dloss / doutput) * (doutput / dx)
        # dloss / doutput就是输入的参数grad_output、
        # 因此只需求relu的导数，在乘以grad_outpu    
        input_, = self.saved_tensors # 把上面保存的input_输出
        if ctx.needs_input_grad[0]:# 判断self.saved_tensors中的Variable是否需要进行反向求导计算梯度
            print('input_ need grad')
        else:
            print('input_ not need grad')
        grad_input = grad_output.clone()
        # 进行梯度计算
        return _se3_matrix_expm_grad(grad_input)


