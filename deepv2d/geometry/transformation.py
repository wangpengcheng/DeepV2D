import tensorflow as tf
import numpy as np

from special_ops.operators import clip_dangerous_gradients
from utils.einsum import einsum

from core.config import cfg
from .se3 import *
from .intrinsics import *
from . import projective_ops as pops
from . import cholesky

cholesky_solve = cholesky.solve


MIN_DEPTH = 0.1
MAX_RESIDUAL = 250.0

# can use both matrix or quaternions to represent rotations
DEFAULT_INTERNAL = cfg.MOTION.INTERNAL


def jac_local_perturb(pt, fill=False):
    X, Y, Z = torch.split(pt, [1, 1, 1], axis=-1)
    o, i = torch.zeros_like(X), torch.ones_like(X)
    if fill:
        j1 = torch.cat([ i,  o,  o, o], axis=-1)
        j2 = torch.cat([ o,  i,  o, o], axis=-1)
        j3 = torch.cat([ o,  o,  i, o], axis=-1)
        j4 = torch.cat([ o, -Z,  Y, o], axis=-1)
        j5 = torch.cat([ Z,  o, -X, o], axis=-1)
        j6 = torch.cat([-Y,  X,  o, o], axis=-1)
    else:
        j1 = torch.cat([ i,  o,  o], axis=-1)
        j2 = torch.cat([ o,  i,  o], axis=-1)
        j3 = torch.cat([ o,  o,  i], axis=-1)
        j4 = torch.cat([ o, -Z,  Y], axis=-1)
        j5 = torch.cat([ Z,  o, -X], axis=-1)
        j6 = torch.cat([-Y,  X,  o], axis=-1)
    jac = torch.stack([j1, j2, j3, j4, j5, j6], axis=-1)
    return jac


def cond_transform(cond, T1, T2):
    """ Return T1 if cond, else T2 """
    
    if T1.internal == 'matrix':
        mat = tf.cond(cond, lambda: T1.matrix(), lambda: T2.matrix())
        T = T1.__class__(matrix=mat, internal=T1.internal)
    
    elif T1.internal == 'quaternion':
        so3 = tf.cond(cond, lambda: T1.so3, lambda: T2.so3)
        translation = tf.cond(cond, lambda: T1.translation, lambda: T2.translation)
        T = T1.__class__(so3=so3, translation=translation, internal=T1.internal)
    
    return T
<<<<<<< HEAD
def stop_gradients(val):
    val.requires_grad = True
def my_shape(val):
    return torch.Tensor(list(val))
def my_transpose(val1,val2):
    return val1.permute(val2)

=======

# SE3初始化
>>>>>>> 317bed8ad3da1341c39a302c231c811b94fb32b7
class SE3:
    def __init__(self, upsilon=None, matrix=None, so3=None, translation=None, eq=None, internal=DEFAULT_INTERNAL):
        self.eq = eq
        self.internal = internal

        if internal == 'matrix':
            if upsilon is not None:
                self.G = se3_matrix_expm(upsilon)
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
            pt = torch.cat([pt, torch.ones_like(pt[...,:1])], axis=-1) # convert to homogenous
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

    def increment(self, upsilon):
        if self.internal == 'matrix':
            G = se3_matrix_increment(self.G, upsilon)
            return self.__class__(matrix=G, internal=self.internal)
        elif self.internal == 'quaternion':
            return self.__class__(upsilon=upsilon, internal=self.internal).__mul__(self)

        
    def concat(self, other, axis=0):
        if self.internal == 'matrix':
            G = torch.cat([self.G, other.G], axis=axis)

        elif self.internal == 'quaternion':
            so3 = torch.cat([self.so3, other.so3], axis=axis)
            t = torch.cat([self.translation, other.translation], axis=axis)
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
<<<<<<< HEAD
        return torch.cat([self.so3, self.translation], axis=-1)
        
=======
        return tf.concat([self.so3, self.translation], axis=-1)
    # 
>>>>>>> 317bed8ad3da1341c39a302c231c811b94fb32b7
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
            torch.cat([A11, A12], axis=-1),
            torch.cat([A21, A22], axis=-1)
        ], axis=-2)

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
            mat = torch.cat([R, t], axis=-1)

            se3_shape = my_shape(self.so3)[:-1]
            filler = torch.tensor([0,0,0,1], dtype=torch.float32)
            filler = torch.repeat(filler[np.newaxis], [torch.prod(se3_shape), 1])
            filler = torch.reshape(filler, torch.cat([se3_shape, [1, 4]], axis=-1))

            if fill:
                mat = torch.cat([mat, filler], axis=-2)

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
<<<<<<< HEAD
        coords0 = pops.coords_grid(my_shape(depth), homogeneous=False)
=======
        coords0 = pops.coords_grid(tf.shape(depth), homogeneous=False) # 获取相机网格坐标
>>>>>>> 317bed8ad3da1341c39a302c231c811b94fb32b7
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

    def fit(self, target, weight, depth, intrinsics, num_iters=1):
        """ minimize geometric reprojection error """
        target = clip_dangerous_gradients(target)
        weight = clip_dangerous_gradients(weight)

        X0 = pops.backproject(depth, intrinsics)
        w = torch.unsqueeze(weight, -1)

        lm_lmbda = cfg.MOTION.LM_LMBDA
        ep_lmbda = cfg.MOTION.EP_LMBDA

        T = EgoSE3Transformation(so3=self.so3, translation=self.translation)
        for i in range(num_iters):
            ### compute the jacobians of the transformation ###
            X1, jtran = T(X0, jacobian=True)
            x1, jproj = pops.project(X1, intrinsics, jacobian=True)

            v = (X0[...,-1] > MIN_DEPTH) &  (X1[...,-1] > MIN_DEPTH)
            v = torch.FloatTensor(v, torch.float32)[..., np, np.newaxis]
            
            ### weighted gauss-newton update ###
            J = einsum('...ij,...jk->...ik', jproj, jtran)
            H = einsum('a...i,a...j->aij', v*w*J, J)
            b = einsum('a...i,a...->ai', v*w*J, target-x1)

            ### add dampening and apply increment ###
            H += (ep_lmbda + lm_lmbda*H)*torch.eye(6)
            delta_upsilon = cholesky_solve(H, b)
            
            dT = EgoSE3Transformation(upsilon=delta_upsilon)
            T = dT * T

        self.so3 = T.so3
        self.translation = T.translation

    def to_dense(self, shape):
        so3 = torch.reshape(self.so3, [-1, 1, 1, 4])
        t = torch.reshape(self.translation, [-1, 1, 1, 3])
        so3 = torch.repeat(so3, [1, shape[0], shape[1], 1])
        t = torch.repeat(t, [1, shape[0], shape[1], 1])
        return DenseSE3Transformation(so3=so3, translation=t)


class VideoSE3Transformation(SE3):
    """ Stores collection of SE3 objects """
    def __init__(self, upsilon=None, matrix=None, so3=None, translation=None, internal=DEFAULT_INTERNAL):
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
            G = torch.gather(self.G, inds, axis=1)
            return VideoSE3Transformation(matrix=G, internal=self.internal)
        elif self.internal == 'quaternion':
            t = torch.gather(self.translation, inds, axis=1)
            so3 = torch.gather(self.so3, inds, axis=1)
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
            G = torch.cat([I, self.G], axis=1)
            return VideoSE3Transformation(matrix=G, internal=self.internal)

        elif self.internal == 'quaternion':
            so3_id = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
            t_id = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

            so3_id = torch.repeat(torch.reshape(so3_id, [1, 1, 4]), [batch, 1, 1])
            t_id = torch.repeat(torch.reshape(t_id, [1, 1, 3]), [batch, 1, 1])

            so3 = torch.cat([so3_id, self.so3], axis=1)
            t = torch.cat([t_id, self.translation], axis=1)
            return VideoSE3Transformation(so3=so3, translation=t, internal=self.internal)

    def keyframe_optim(self, 
                       target, 
                       weight, 
                       depth, 
                       intrinsics, 
                       num_iters=2):
        """ Motion only BA using keyframe
            targets: dense correspondence field, maps points in I(i) -> I(j)
            weights: confidence of the correspondence vectors in range [0,1]
            depths: depths for the input frames
            intrinsics: 3x3 intrinsics matrix
            include_intrinsics: include focal length in optimization
            num_iters: number of Gauss-Newton iterations
            num_fixed: number of camera poses to set as fixed
        """

        target = clip_dangerous_gradients(target)
        weight = clip_dangerous_gradients(weight)

        X0 = pops.backproject(depth, intrinsics)
        w = torch.unsqueeze(weight, axis=-1)

        lm_lmbda = cfg.MOTION.LM_LMBDA
        ep_lmbda = cfg.MOTION.EP_LMBDA

        T = self.copy(stop_gradients=False)
        for i in range(num_iters):
            ### compute the jacobians of the transformation ###
            X1, jtran = T(X0, jacobian=True)
            x1, (jproj, jkvec) = pops.project(X1, intrinsics, jacobian=True)

            v = (X0[...,-1] > MIN_DEPTH) & (X1[...,-1] > MIN_DEPTH)
            v = torch.FloatTensor(v, torch.float32)[..., np.newaxis, np.newaxis]
            
            ### weighted gauss-newton update ###
            J = einsum('...ij,...jk->...ik', jproj, jtran)
            #tf\.add_to_collection("checkpoints", J)

            H = einsum('ai...j,ai...k->aijk', v*w*J, J)
            b = einsum('ai...j,ai...->aij', v*w*J, target-x1)

            #tf\.add_to_collection("checkpoints", H)
            #tf\.add_to_collection("checkpoints", b)

            ### add dampening and apply increment ###
            H += ep_lmbda*torch.eye(6) + lm_lmbda*H*torch.eye(6)
            delta_upsilon = cholesky_solve(H, b)
            #tf\.add_to_collection("checkpoints", delta_upsilon)

            T = T.increment(delta_upsilon)

        # update
        if self.internal == 'matrix':
            self.G = T.matrix()
            T = VideoSE3Transformation(matrix=T.matrix(), internal=self.internal)

        elif self.internal == 'quaternion':
            self.so3 = T.so3
            self.translation = T.translation
            T = VideoSE3Transformation(so3=self.so3, translation=self.translation, internal=self.internal)

        return T


    def global_optim(self, 
                     targets, 
                     weights, 
                     depths, 
                     intrinsics, 
                     inds,
                     include_intrinsics=False,
                     num_iters=1, 
                     num_fixed=0):
        """ Motion only BA in the general case
            targets: dense correspondence field, maps points in I(i) -> I(j)
            weights: confidence of the correspondence vectors in range [0,1]
            depths: depths for the input frames
            intrinsics: 3x3 intrinsics matrix
            inds: edges in the graph; (i, j) indicates flow observation from i->j
            include_intrinsics: include focal length in optimization
            num_iters: number of Gauss-Newton iterations
            num_fixed: number of camera poses to set as fixed
        """
        
        # damping
        lm_lmbda = cfg.MOTION.LM_LMBDA
        ep_lmbda = cfg.MOTION.EP_LMBDA

        batch, num = self.shape()
        dof = 6
        ii = inds[0]
        jj = inds[1]

        targets = clip_dangerous_gradients(targets)
        weights = clip_dangerous_gradients(weights)
        w = torch.unsqueeze(weights, -1)

        T = self.copy(stop_gradients=False)
        for i in range(num_iters):

            ### relative transformation between pairs ###
            Tij = T.gather(ii) * T.gather(jj).inv()
            
            ### compute the jacobians of the transformation ###
            X0, jkvec1 = pops.backproject(depths, intrinsics, jacobian=True)
            X1, jtran = Tij(X0, jacobian=True)
            x1, (jproj, jkvec2) = pops.project(X1, intrinsics, jacobian=True)

            residual_mag = torch.sqrt(torch.sum((targets - x1)**2, axis=-1))
            
            v = (X0[...,-1] > MIN_DEPTH) & \
                (X1[...,-1] > MIN_DEPTH) & \
                (residual_mag < MAX_RESIDUAL)

            v = torch.FloatTensor(v, torch.float32)[..., np.newaxis, np.newaxis]

            # jacobians linearly related through adjoint
            Ji = einsum("...ij,...jk->...ik", jproj, jtran)
            Jj = -einsum("ab...ij,abjk->ab...ik", Ji, Tij.adj())
            
            # jacobians of intrinsics
            jkvec1 = einsum('abij,ab...jk->ab...ik', Tij.matrix(), jkvec1)
            Jk = jkvec2 + einsum('...ij,...jk->...ik', jproj, jkvec1[...,:3,:])

            f = (intrinsics[:, 0,0] + intrinsics[:, 1,1]) / 2.0
            Jk = einsum('a...,a->a...', Jk, f)

            #tf\.add_to_collection("checkpoints", Ji)
            #tf\.add_to_collection("checkpoints", Jj)
            #tf\.add_to_collection("checkpoints", Jk)

            ### build the system Hx = b ###
            H11_ii = einsum('ai...j,ai...k->iajk', v*w*Ji, Ji)
            H11_ij = einsum('ai...j,ai...k->iajk', v*w*Ji, Jj)
            H11_ji = einsum('ai...j,ai...k->iajk', v*w*Jj, Ji)
            H11_jj = einsum('ai...j,ai...k->iajk', v*w*Jj, Jj)

            h11_shape = [num, num, batch, dof, dof]
            H11 = tf.scatter_nd(torch.stack([ii,ii], axis=-1), H11_ii, h11_shape) + \
                  tf.scatter_nd(torch.stack([ii,jj], axis=-1), H11_ij, h11_shape) + \
                  tf.scatter_nd(torch.stack([jj,ii], axis=-1), H11_ji, h11_shape) + \
                  tf.scatter_nd(torch.stack([jj,jj], axis=-1), H11_jj, h11_shape)

            b1_i = einsum('ai...j,ai...->iaj', v*w*Ji, targets-x1)
            b1_j = einsum('ai...j,ai...->iaj', v*w*Jj, targets-x1)

            b_shape = [num, batch, dof]
            b1 =  tf.scatter_nd(torch.stack([ii], axis=-1), b1_i, b_shape) + \
                  tf.scatter_nd(torch.stack([jj], axis=-1), b1_j, b_shape)

            H11 = torch.reshape(my_transpose(H11, [2,0,3,1,4]), [batch, num*dof, num*dof])
            H11 += (ep_lmbda + lm_lmbda*H11)*torch.eye(dof*num)

            b1 = torch.reshape(my_transpose(b1, [1,0,2]), [batch, num*dof])

            # keep intrinsics matrix fixed
            if not include_intrinsics:
                H = H11
                b = b1

            else:
                # Builds the system (H11 pose block, H22 intrinsics block H12: pose-intrinsics block
                # H = [H11, H12]
                #     [H21, H22]

                H12_i = einsum('ai...j,ai...k->iajk', v*w*Ji, Jk)
                H12_j = einsum('ai...j,ai...k->iajk', v*w*Jj, Jk)

                h12_shape = [num, 1, batch, dof, 1]
                H12 = tf.scatter_nd(torch.stack([ii, torch.zeros_like(ii)], axis=-1), H12_i, h12_shape) + \
                    tf.scatter_nd(torch.stack([jj, torch.zeros_like(jj)], axis=-1), H12_j, h12_shape)

                H22 = einsum('ai...j,ai...k->ajk', v*w*Jk, Jk) 
                b2 = einsum('a...j,a...->aj', v*w*Jk, targets-x1)

                H12 = torch.reshape(my_transpose(H12, [2,0,3,1,4]), [batch, num*dof, 1])
                H21 = my_transpose(H12, [0, 2, 1])
                H22 = torch.reshape(H22, [batch, 1, 1])

                flm_lmbda = 0.001 # damping for focal length update
                fep_lmbda = 100.0
                H22 += (flm_lmbda * H22 + fep_lmbda) * torch.eye(1)

                H = torch.cat([
                    torch.cat([H11, H12], axis=-1),
                    torch.cat([H21, H22], axis=-1)], axis=-2)

                b = torch.cat([b1, b2], axis=-1)
                
            # system is built, now solve it
            #tf\.add_to_collection("checkpoints", H)
            #tf\.add_to_collection("checkpoints", b)

            num_free = num - num_fixed
            H = H[:, dof*num_fixed:, dof*num_fixed:]
            b = b[:, dof*num_fixed:]

            if include_intrinsics:
                delta_update = cholesky_solve(H, b)
                #tf\.add_to_collection("checkpoints", delta_update)
                delta_upsilon, delta_intrinsics = torch.split(delta_update, [num_free*dof, 1], axis=-1)
                intrinsics = update_intrinsics(intrinsics, delta_intrinsics)
           
            else:
                delta_upsilon = cholesky_solve(H, b)
                #tf\.add_to_collection("checkpoints", delta_upsilon)

            delta_upsilon = torch.reshape(delta_upsilon, [batch, num_free, dof])
            zeros_upsilon = torch.zeros([batch, num_fixed, dof])
            delta_upsilon = torch.cat([zeros_upsilon, delta_upsilon], axis=1)
            T = T.increment(delta_upsilon)

        # update
        if self.internal == 'matrix':
            self.G = T.matrix()
            T = VideoSE3Transformation(matrix=T.matrix(), internal=self.internal)

        elif self.internal == 'quaternion':
            self.so3 = T.so3
            self.translation = T.translation
            T = VideoSE3Transformation(so3=self.so3, translation=self.translation, internal=self.internal)

        return T, intrinsics

    def transform(self, depth, intrinsics, valid_mask=False, return3d=False):
        return super(VideoSE3Transformation, self).transform(depth, intrinsics, valid_mask, return3d)
