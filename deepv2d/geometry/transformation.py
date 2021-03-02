import numpy as np

from utils.einsum import einsum

from core.config import cfg
from .se3 import *
from .intrinsics import *
from . import projective_ops as pops
#from . import cholesky

#cholesky_solve = cholesky.solve
def my_gather(input, indexs,dim=1):
    return input.index_select(dim, indexs)

MIN_DEPTH = 0.1
MAX_RESIDUAL = 250.0

# can use both matrix or quaternions to represent rotations
DEFAULT_INTERNAL = cfg.MOTION.INTERNAL


def jac_local_perturb(pt, fill=False):
    X, Y, Z = torch.split(pt, [1, 1, 1], dim=-1)
    o, i = torch.zeros_like(X), torch.ones_like(X)
    if fill:
        j1 = torch.cat([ i,  o,  o, o], dim=-1)
        j2 = torch.cat([ o,  i,  o, o], dim=-1)
        j3 = torch.cat([ o,  o,  i, o], dim=-1)
        j4 = torch.cat([ o, -Z,  Y, o], dim=-1)
        j5 = torch.cat([ Z,  o, -X, o], dim=-1)
        j6 = torch.cat([-Y,  X,  o, o], dim=-1)
    else:
        j1 = torch.cat([ i,  o,  o], dim=-1)
        j2 = torch.cat([ o,  i,  o], dim=-1)
        j3 = torch.cat([ o,  o,  i], dim=-1)
        j4 = torch.cat([ o, -Z,  Y], dim=-1)
        j5 = torch.cat([ Z,  o, -X], dim=-1)
        j6 = torch.cat([-Y,  X,  o], dim=-1)
    jac = torch.stack([j1, j2, j3, j4, j5, j6], dim=-1)
    return jac


# def cond_transform(cond, T1, T2):
#     """ Return T1 if cond, else T2 """
    
#     if T1.internal == 'matrix':
#         mat = tf.cond(cond, lambda: T1.matrix(), lambda: T2.matrix())
#         T = T1.__class__(matrix=mat, internal=T1.internal)
    
#     elif T1.internal == 'quaternion':
#         so3 = tf.cond(cond, lambda: T1.so3, lambda: T2.so3)
#         translation = tf.cond(cond, lambda: T1.translation, lambda: T2.translation)
#         T = T1.__class__(so3=so3, translation=translation, internal=T1.internal)
    
#     return T

def stop_gradients(val):
    val.requires_grad = False
def my_shape(val):
    return torch.Tensor(list(val))
def my_transpose(val1, val2):
    return val1.permute(val2)


# SE3初始化
class SE3:
    def __init__(
                self, 
                upsilon=None, 
                matrix=None, 
                so3=None, 
                translation=None, 
                eq=None, 
                internal=DEFAULT_INTERNAL
                ):
        self.eq = eq
        self.internal = internal
        self.se3_matrix_expm = Se3MatrixExpm()

        if internal == 'matrix':
            if upsilon is not None:
                self.G = self.se3_matrix_expm(upsilon)
            elif matrix is not None:
                self.G = matrix

        elif internal == 'quaternion':
            if upsilon is not None:
                self.so3, self.translation = se3_expm(upsilon)
            elif matrix is not None:
                R, t = matrix[...,:3,:3], matrix[...,:3,3]
                self.so3 = rotation_matrix_to_quaternion(R)
                self.translation = t
            elif (so3 is not None) and (translation is not None):
                self.so3 = so3
                self.translation = translation

    def __call__(self, pt, jacobian=False):
        """ Transform set of points """

        if self.internal == 'matrix':
            pt = torch.cat([pt, torch.ones_like(pt[...,:1])], dim=-1) # convert to homogenous
            pt = einsum(self.eq, self.G[..., :3, :], pt)
        
        elif self.internal == 'quaternion':
            pt = quaternion_rotate_point(self.so3, pt, self.eq)
            pt = pt + self.translation

        if jacobian:
            jacobian = jac_local_perturb(pt)
            return pt, jacobian

        return pt

    def __mul__(self, other):
        if self.internal == 'matrix':
            G = torch.matmul(self.G, other.G)
            return self.__class__(matrix=G, internal=self.internal)

        elif self.internal == 'quaternion':
            so3 = quaternion_multiply(self.so3, other.so3)
            translation = self.translation + quaternion_rotate_point(self.so3, other.translation)
            return self.__class__(so3=so3, translation=translation, internal=self.internal)


        
    def concat(self, other, dim=0):
        if self.internal == 'matrix':
            G = torch.cat([self.G, other.G], dim=axis)

        elif self.internal == 'quaternion':
            so3 = torch.cat([self.so3, other.so3], dim=axis)
            t = torch.cat([self.translation, other.translation], dim=axis)
            return self.__class__(so3=so3, translation=t, internal=self.internal)

    def copy(self, stop_gradients=False):

        if self.internal == 'matrix':
            if stop_gradients:
                return self.__class__(matrix=stop_gradient(self.G), internal=self.internal)
            else:
                return self.__class__(matrix=self.G, internal=self.internal)

        elif self.internal == 'quaternion':
            if stop_gradients:
                so3 = self.so3
                t = stop_gradient(self.translation)
                return self.__class__(so3=so3, translation=t, internal=self.internal)
            else:
                return self.__class__(so3=self.so3, translation=self.translation, internal=self.internal)

    def to_vec(self):
        return torch.cat([self.so3, self.translation], dim=-1)
        
    def inv(self):
        if self.internal == 'matrix':
            Ginv = se3_matrix_inverse(self.matrix())
            return self.__class__(matrix=Ginv, internal=self.internal)
        elif self.internal == 'quaternion':
            inv_so3 = quaternion_inverse(self.so3)
            inv_translation = quaternion_rotate_point(inv_so3, -self.translation)
            return self.__class__(so3=inv_so3, translation=inv_translation, internal=self.internal)

    def adj(self):
        if self.internal == 'matrix':
            R = self.G[..., :3, :3]
            t = self.G[..., :3, 3]
            A11 = R
            A12 = torch.matmul(hat(t), R)
            A21 = torch.zeros_like(A11)
            A22 = R

        elif self.internal == 'quaternion':
            A11 = quaternion_to_matrix(self.so3)
            A12 = torch.matmul(hat(self.translation), A11)
            A21 = torch.zeros_like(A11)
            A22 = quaternion_to_matrix(self.so3)

        Ax = torch.cat([
            torch.cat([A11, A12], dim=-1),
            torch.cat([A21, A22], dim=-1)
        ], dim=-2)

        return Ax

    def logm(self):            
        return se3_logm(self.so3, self.translation)

    def shape(self):
        return (self.so3.shape)[:-1]

    def matrix(self, fill=True):
        if self.internal == 'matrix':
            return self.G
        elif self.internal == 'quaternion':
            R = quaternion_to_matrix(self.so3)
            t = torch.unsqueeze(self.translation,-1)
            mat = torch.cat([R, t], dim=-1)

            se3_shape = my_shape(self.so3)[:-1]
            filler = torch.tensor([0,0,0,1], dtype=torch.float32)
            filler = torch.repeat(filler[np.newaxis], [torch.prod(se3_shape), 1])
            filler = torch.reshape(filler, torch.cat([se3_shape, [1, 4]], dim=-1))

            if fill:
                mat = torch.cat([mat, filler], dim=-2)

            return mat
    # 将深度图像，转换为(X，Y,Z)点云图
    def transform(self, depth, intrinsics, valid_mask=False, return3d=False):
        pt = pops.backproject(depth, intrinsics) # 根据深度和相机内参获取三维点云
        pt_new = self.__call__(pt) # 获取新的三维点云图像
        coords = pops.project(pt_new, intrinsics)
        if return3d: 
            return coords, pt_new
        if valid_mask:
            vmask = (pt[...,-1] > MIN_DEPTH) & (pt_new[...,-1] > MIN_DEPTH)
            vmask = torch.FloatTensor(vmask, torch.float32)[..., np.newaxis]
            return coords, vmask
        return coords
    # 特征相机网络
    def induced_flow(self, depth, intrinsics, valid_mask=False):
        coords0 = pops.coords_grid(my_shape(depth), homogeneous=False)
        if valid_mask:
            coords1, vmask = self.transform(depth, intrinsics, valid_mask=valid_mask)
            return coords1 - coords0, vmask
        coords1 = self.transform(depth, intrinsics, valid_mask=valid_mask)
        return coords1 - coords0 # 获取相机坐标差值

    def depth_change(self, depth, intrinsics):
        pt = pops.backproject(depth, intrinsics)
        pt_new = self.__call__(pt)
        return pt_new[...,-1] - pt[...,-1] 


class EgoSE3Transformation(SE3):
    """ Ego transformation mapping """
    def __init__(self, upsilon=None, matrix=None, so3=None, translation=None):
        super(EgoSE3Transformation, self).__init__(upsilon, matrix, so3, translation)

    def __call__(self, pt, jacobian=False):
        t = self.translation[:, np.newaxis, np.newaxis]
        return SE3(so3=self.so3, translation=t, eq='aij,a...j->a...i')(pt, jacobian=jacobian)

    # def fit(self, target, weight, depth, intrinsics, num_iters=1):
    #     """ minimize geometric reprojection error """
    #     target = clip_dangerous_gradients(target)
    #     weight = clip_dangerous_gradients(weight)

    #     X0 = pops.backproject(depth, intrinsics)
    #     w = torch.unsqueeze(weight, -1)

    #     lm_lmbda = cfg.MOTION.LM_LMBDA
    #     ep_lmbda = cfg.MOTION.EP_LMBDA

    #     T = EgoSE3Transformation(so3=self.so3, translation=self.translation)
    #     for i in range(num_iters):
    #         ### compute the jacobians of the transformation ###
    #         X1, jtran = T(X0, jacobian=True)
    #         x1, jproj = pops.project(X1, intrinsics, jacobian=True)

    #         v = (X0[...,-1] > MIN_DEPTH) &  (X1[...,-1] > MIN_DEPTH)
    #         v = torch.FloatTensor(v, torch.float32)[..., np, np.newaxis]
            
    #         ### weighted gauss-newton update ###
    #         J = einsum('...ij,...jk->...ik', jproj, jtran)
    #         H = einsum('a...i,a...j->aij', v*w*J, J)
    #         b = einsum('a...i,a...->ai', v*w*J, target-x1)

    #         ### add dampening and apply increment ###
    #         H += (ep_lmbda + lm_lmbda*H)*torch.eye(6)
    #         # 计算梯度
    #         delta_upsilon = cholesky_solve(H, b)
            
    #         dT = EgoSE3Transformation(upsilon=delta_upsilon)
    #         T = dT * T

    #     self.so3 = T.so3
    #     self.translation = T.translation

    def to_dense(self, shape):
        so3 = torch.reshape(self.so3, [-1, 1, 1, 4])
        t = torch.reshape(self.translation, [-1, 1, 1, 3])
        so3 = torch.repeat(so3, [1, shape[0], shape[1], 1])
        t = torch.repeat(t, [1, shape[0], shape[1], 1])
        return DenseSE3Transformation(so3=so3, translation=t)


class VideoSE3Transformation(SE3):
    """ Stores collection of SE3 objects """
    def __init__(
                self, 
                upsilon=None, 
                matrix=None, 
                so3=None, 
                translation=None, 
                internal=DEFAULT_INTERNAL
                ):
        super(VideoSE3Transformation, self).__init__(upsilon, matrix, so3, translation, internal=internal)
        self.eq = "aijk,ai...k->ai...j"

    def __call__(self, pt, inds=None, jacobian=False):
        if self.internal == 'matrix':
            return super(VideoSE3Transformation, self).__call__(pt, jacobian=jacobian)
        elif self.internal == 'quaternion':
            ndim = len(pt.get_shape().as_list())
            t = self.translation
            for i in range(ndim-3):
                t = t[:, :, np.newaxis]
            return SE3(so3=self.so3, translation=t, eq="aijk,ai...k->ai...j")(pt, jacobian=jacobian)

    def gather(self, inds):
        if self.internal == 'matrix':
            G = my_gather(self.G, inds, dim=1)
            return VideoSE3Transformation(matrix=G, internal=self.internal)
        elif self.internal == 'quaternion':
            t = my_gather(self.translation, inds, dim=1)
            so3 = my_gather(self.so3, inds, dim=1)
            return VideoSE3Transformation(so3=so3, translation=t, internal=self.internal)

    def shape(self):
        if self.internal == 'matrix':
            my_shape = my_shape(self.G)
        elif self.internal == 'quaternion':
            my_shape = my_shape(self.so3)
        
        return (my_shape[0], my_shape[1])

    def append_identity(self):
        """ Push identity transformation to start of collection """
        batch, frames = self.shape()
        if self.internal == 'matrix':
            I = torch.eye(4, batch_shape=[batch, 1])
            G = torch.cat([I, self.G], dim=1)
            return VideoSE3Transformation(matrix=G, internal=self.internal)

        elif self.internal == 'quaternion':
            so3_id = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
            t_id = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

            so3_id = torch.repeat(torch.reshape(so3_id, [1, 1, 4]), [batch, 1, 1])
            t_id = torch.repeat(torch.reshape(t_id, [1, 1, 3]), [batch, 1, 1])

            so3 = torch.cat([so3_id, self.so3], dim=1)
            t = torch.cat([t_id, self.translation], dim=1)
            return VideoSE3Transformation(so3=so3, translation=t, internal=self.internal)


    def transform(self, depth, intrinsics, valid_mask=False, return3d=False):
        return super(VideoSE3Transformation, self).transform(depth, intrinsics, valid_mask, return3d)
