import tensorflow as tf
import numpy as np
import time
import cv2
import vis
from scipy import interpolate
import matplotlib.pyplot as plt

from modules.depth import DepthNetwork
from modules.motion import MotionNetwork

from utils import flow_viz
from fcrn import fcrn
from geometry.intrinsics import *
from geometry.transformation import *
from geometry import projective_ops

# 填充深度图像，主要是对深度图像进行线性插值
def fill_depth(depth):
    """ Fill in the holes in the depth map  将整个图填充为深度图像
    """
    ht, wd = depth.shape
    x, y = np.meshgrid(np.arange(wd), np.arange(ht))
    xx = x[depth > 0].astype(np.float32)
    yy = y[depth > 0].astype(np.float32)
    zz = depth[depth > 0].ravel()
    # 进行插值 
    return interpolate.griddata((xx, yy), zz, (x, y), method='linear')

# 将旋转转换为角度
def vee(R):
    x1 = R[2,1] - R[1,2]
    x2 = R[0,2] - R[2,0]
    x3 = R[1,0] - R[0,1]
    return np.array([x1, x2, x3])

# 获取位姿距离；主要用来表示变换的剧烈程度
def pose_distance(G):
    # 获取相机旋转R和平移t 
    R, t = G[:3,:3], G[:3,3]
    r = vee(R)
    # dr的均方和
    dR = np.sqrt(np.sum(r**2))
    dt = np.sqrt(np.sum(t**2))
    return dR, dt


class DeepV2D:
    """
    推理测试主要类
    """
    def __init__(self, 
                 cfg,
                 ckpt, 
                 is_calibrated=True, 
                 use_fcrn=False, 
                 use_regressor=True, 
                 image_dims=None,
                 mode='keyframe'
                 ):

        self.cfg = cfg # cfg配置文件
        self.ckpt = ckpt # ckpt文件位置
        self.mode = mode # 加载的模型文件

        self.use_fcrn = use_fcrn # 使用文件
        self.use_regressor = use_regressor # 用户
        self.is_calibrated = is_calibrated # 是否已经校准

        if image_dims is not None:
            self.image_dims = image_dims
        else:
            if cfg.STRUCTURE.MODE == 'concat':
                self.image_dims = [cfg.INPUT.FRAMES, cfg.INPUT.HEIGHT, cfg.INPUT.WIDTH]
            else:
                self.image_dims = [None, int(cfg.INPUT.HEIGHT*cfg.INPUT.RESIZE), (cfg.INPUT.WIDTH*cfg.INPUT.RESIZE)] # 构建输入的训练图片维度

        self.outputs = {}
        # 创建预定于变量
        self._create_placeholders()
        # 创建位姿估计网络
        #self._build_motion_graph()
        # 创建深度网络
        self._build_depth_graph()
        # 构建深度映射图
        self._build_reprojection_graph()
        # 相似值图
        self._build_visibility_graph()
        # 创建点云图
        self._build_point_cloud_graph()
        # 深度
        self.depths = []
        # 位姿
        self.poses = []

        if self.use_fcrn:
            self._build_fcrn_graph() # 
        # 加载模型
        self.saver = tf.train.Saver(tf.model_variables()) #构建存储模型

    # 创建session
    def set_session(self, sess):
        self.sess = sess
        sess.run(tf.global_variables_initializer())
        # 存储
        self.saver.restore(self.sess, self.ckpt)

        if self.use_fcrn:
            fcrn_vars = {}
            for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="FCRN"):
                fcrn_vars[var.name.replace('FCRN/', '').replace(':0', '')] = var

            fcrn_saver = tf.train.Saver(fcrn_vars)
            fcrn_saver.restore(sess, 'models/NYU_FCRN.ckpt')


    def _create_placeholders(self):
        """
        创建预定义变量；主要是构建数据
        """
        frames, ht, wd = self.image_dims
        # 输入数据占位，注意这里的输入是3张图片
        self.images_placeholder = tf.placeholder(tf.float32, [frames, ht, wd, 3])
        # 如果是关键帧模式frames值为1
        if self.mode == 'keyframe':
            self.depths_placeholder = tf.placeholder(tf.float32, [1, ht, wd])
        else:
            self.depths_placeholder = tf.placeholder(tf.float32, [frames, ht, wd])
        # 位姿内存分配；每一个位姿都是4*4的一个矩阵
        self.poses_placeholder = tf.placeholder(tf.float32, [frames, 4, 4])
        # 相机内参矩阵一维数组
        self.intrinsics_placeholder = tf.placeholder(tf.float32, [4])
        self.init_placeholder = tf.placeholder(tf.bool, []) # bool 数据是否占位符号

        # placeholders for storing graph adj_list and edges
        
        self.edges_placeholder = tf.placeholder(tf.int32, [None, 2]) # 边缘函数
        self.adj_placeholder = tf.placeholder(tf.int32, [None, None]) # adj函数
    # 构建位姿估计网络
    def _build_motion_graph(self):
        self.motion_net = MotionNetwork(self.cfg.MOTION, mode=self.mode,
            use_regressor=self.use_regressor, is_calibrated=self.is_calibrated, is_training=False)
        # 定义图相关数据，注意这里的数据是共享的
        images = self.images_placeholder[tf.newaxis] # 输入图像数据 
        depths = self.depths_placeholder[tf.newaxis] # 定义深度数据
        poses = self.poses_placeholder[tf.newaxis] # 定义相机姿态数据
        
        do_init = self.init_placeholder # 初始化数据
        intrinsics = self.intrinsics_placeholder[tf.newaxis] # 定义相机位姿
        edge_inds = tf.unstack(self.edges_placeholder, num=2, axis=-1) # 在这里进行矩阵分解
        # 将位姿矩阵，转换位SE3
        # convert pose matrix into SE3 object
        Ts = VideoSE3Transformation(matrix=poses)
        # 线进行位姿初始化，并进行前向计算
        Ts, intrinsics = self.motion_net.forward(
            Ts, 
            images, 
            depths, 
            intrinsics, 
            edge_inds, 
            init=do_init
            )
        # 更新位姿信息
        self.outputs['poses'] = tf.squeeze(Ts.matrix(), 0)
        # 更新相机参数
        self.outputs['intrinsics'] = intrinsics[0]
        # 更新权重
        self.outputs['weights'] = self.motion_net.weights_history[-1]

    def _build_depth_graph(self):
        """
        构建深度图信息
        """
        self.depth_net = DepthNetwork(self.cfg.STRUCTURE, is_training=False)
        images = self.images_placeholder[tf.newaxis]
        poses = self.poses_placeholder[tf.newaxis]
        intrinsics = self.intrinsics_placeholder[tf.newaxis]

        # convert pose matrix into SE3 object
        Ts = VideoSE3Transformation(matrix=poses)

        adj_list = None
        if self.mode == 'global':
            adj_list = self.adj_placeholder
        
        depths = self.depth_net.forward(
            Ts, # 相机位姿 
            images, # 图像 
            intrinsics, # 相机参数
            adj_list # 激活函数
            )
        # 更新深度图
        self.outputs['depths'] = depths
        self.outputs['poses'] = tf.squeeze(Ts.matrix(), 0)
        self.outputs['intrinsics'] = intrinsics[0]

    # 创建点云图
    def _build_point_cloud_graph(self):
        """Use poses and depth maps to create point cloud"""
        depths = self.depths_placeholder[tf.newaxis]
        images = self.images_placeholder[tf.newaxis]
        poses = self.poses_placeholder[tf.newaxis]
        intrinsics = self.intrinsics_placeholder[tf.newaxis]
        intrinsics = intrinsics_vec_to_matrix(intrinsics)

        depths_pad = tf.pad(depths, [[0,0],[0,0],[0,1],[0,1]], "CONSTANT")

        depths_grad = \
            (depths_pad[:, :, 1:, :-1] - depths_pad[:, :, :-1, :-1])**2 + \
            (depths_pad[:, :, :-1, 1:] - depths_pad[:, :, :-1, :-1])**2

        # don't use large depths for point cloud and ignore boundary regions
        valid = (depths < 5.0) & (depths_grad < 0.01)

        # depths, intrinsics = rescale_depths_and_intrinsics(depths, intrinsics, downscale=4)
        batch, num, ht, wd = tf.unstack(tf.shape(depths), num=4)

        ii, jj = tf.meshgrid(tf.range(1), tf.range(0, num))
        ii = tf.reshape(ii, [-1]) # 将其变为一维度向量
        jj = tf.reshape(jj, [-1])

        Ts = VideoSE3Transformation(matrix=poses)
        X0 = projective_ops.backproject(depths, intrinsics)
        
        # transform point cloud into coordinate system defined by first frame
        X1 = (Ts.gather(ii) * Ts.gather(jj).inv())(X0)

        crop_h = 12
        crop_w = 32

        X1 = X1[:, :, crop_h:-crop_h, crop_w:-crop_w]
        valid = valid[:, :, crop_h:-crop_h, crop_w:-crop_w]
        images = images[:, :, crop_h:-crop_h, crop_w:-crop_w, ::-1]
        
        X1 = tf.reshape(X1, [-1, 3]) #n*3个三维向量
        colors = tf.reshape(images, [-1, 3])

        valid_inds = tf.where(tf.reshape(valid, [-1]))
        valid_inds = tf.reshape(valid_inds, [-1])

        X1 = tf.gather(X1, valid_inds, axis=0)
        colors = tf.gather(colors, valid_inds, axis=0)

        self.outputs['point_cloud'] = (X1, colors)


    def _build_reprojection_graph(self):
        """ Used to project depth from keyframes onto new frame 用于将深度从关键帧投影到新帧上"""

        EPS = 1e-8
        depths = self.depths_placeholder[tf.newaxis]
        poses = self.poses_placeholder[tf.newaxis]
        intrinsics = self.intrinsics_placeholder[tf.newaxis]

        batch, num, ht, wd = tf.unstack(tf.shape(depths), num=4)
        Ts = VideoSE3Transformation(matrix=poses)
        intrinsics = intrinsics_vec_to_matrix(intrinsics) # 将相机参数转换为矩阵

        ii, jj = tf.meshgrid(tf.range(0, num), tf.range(num, num+1)) # 构建多帧的数量，ii为num,jj为1
        ii = tf.reshape(ii, [-1]) # 将其转换为1维度向量
        jj = tf.reshape(jj, [-1]) # 

        Tij = Ts.gather(jj) * Ts.gather(ii).inv() # 获取对应的照片位姿i*j;注意j一般为1，进行取反
        X0 = projective_ops.backproject(depths, intrinsics) # 获取点云图
        X1 = Tij(X0) # 获取对应点的位姿，注意这里的点应该是n,w,h,x,y,d的6维度数据

        coords = projective_ops.project(X1, intrinsics) # 获取每个点对应的x,y
        depths = X1[..., 2] #获取深度图

        indicies = tf.cast(coords[..., ::-1] + .5, tf.int32) #
        indicies = tf.reshape(indicies, [-1, 2]) # 转换为2列数据，主要是X,Y
        depths = tf.reshape(depths, [-1]) # 将深度数据转换为一个数组

        depth = tf.scatter_nd(indicies, depths, [ht, wd]) # 转换为ht*wd的二维矩阵，主要值是depth
        count = tf.scatter_nd(indicies, tf.ones_like(depths), [ht, wd]) # 将其分散到新的张量中,来统计是否有深度数据

        depth = depth / (count + EPS) #在这里对深度进行均一化
        self.outputs['depth_reprojection'] = depth # 获取深度映射信息

    def _build_visibility_graph(self):
        """ Find induced optical flow between pairs of frames 查找帧之间产生的光流 """

        depths = self.depths_placeholder[tf.newaxis] # 深度
        poses = self.poses_placeholder[tf.newaxis] # 位姿
        intrinsics = self.intrinsics_placeholder[tf.newaxis] # 相机内参

        Ts = VideoSE3Transformation(matrix=poses) # 将位姿转换为矩阵
        ii, jj = tf.unstack(self.edges_placeholder, num=2, axis=-1) # 这里将边缘矩阵进分解，-1表示逆序的最后一个维度
        intrinsics = intrinsics_vec_to_matrix(intrinsics) # 进行参数转换
        # 缩放深度信息和特征点信息
        depths, intrinsics = rescale_depths_and_intrinsics(depths, intrinsics, downscale=4)
        ht = tf.cast(tf.shape(depths)[2], tf.float32)
        wd = tf.cast(tf.shape(depths)[3], tf.float32)

        depths = tf.gather(depths, ii, axis=1) # 从depths的axis维根据ii的参数值获取切片
        Tij = Ts.gather(jj) * Ts.gather(ii).inv() # 进行位姿数据切片

        flow = Tij.induced_flow(depths, intrinsics) # 计算流
        coords = Tij.transform(depths, intrinsics) # 进行坐标转换

        flo_graph = tf.sqrt(tf.reduce_sum(flow**2, axis=-1))
        flo_graph = tf.reduce_mean(flo_graph, [-1, -2])

        contained = tf.to_float(
            (coords[...,0] > 0.0) & (coords[...,0] < wd) & 
            (coords[...,1] > 0.0) & (coords[...,1] < ht))
        # 在这里进行求取中间值
        vis_graph = tf.reduce_mean(contained, [-1, -2])
        self.outputs['visibility'] = (flo_graph[0], vis_graph[0], flow) # 可见性

    def _build_fcrn_graph(self):
        """ Build single image initializion graph"""
        images = self.images_placeholder
        batch, ht, wd, _ = tf.unstack(tf.shape(images), num=4)

        with tf.variable_scope("FCRN") as scope:
            # crop out boarder and flip color channels
            fcrn_input = tf.image.resize_area(images[:, 4:-4, 6:-6, ::-1], [228, 304])
            net = fcrn.ResNet50UpProj({'data': fcrn_input}, batch, 1, False)
            fcrn_output = tf.stop_gradient(net.get_output())
            fcrn_output = tf.image.resize_bilinear(fcrn_output, [ht, wd])

        self.outputs['fcrn'] = tf.squeeze(fcrn_output, -1)

    def compute_visibility_matrix(self):
        """ Computes a matrix of optical flow and visibility between all pairs of frames 
        Ex. flo_matrix[i,j] is the mean optical flow between camera i and camera j
        Ex. vis_matrix[i,j] is the portion of points in camera i visibile in camera j """
        
        num = len(self.images)
        ii, jj = np.meshgrid(np.arange(num), np.arange(num))
        
        ii = np.reshape(ii, [-1])
        jj = np.reshape(jj, [-1])
        edges = np.stack([jj, ii], axis=-1)

        feed_dict = {
            self.depths_placeholder: self.depths,
            self.poses_placeholder: self.poses,
            self.edges_placeholder: edges,
            self.intrinsics_placeholder: self.intrinsics}

        flo_graph, vis_graph, flow = self.sess.run(self.outputs['visibility'], feed_dict=feed_dict)
        flo_matrix = flo_graph.reshape(num, num)
        vis_matrix = vis_graph.reshape(num, num)
        return flo_matrix, vis_matrix, flow

    def reproject_depth(self, query_pose):
        """ Use depth estimates and poses to estimate depth map at a new camera location """
        poses = np.concatenate([self.poses, query_pose[np.newaxis]], axis=0)
        feed_dict = {
            self.depths_placeholder: self.depths,
            self.poses_placeholder: poses,
            self.intrinsics_placeholder: self.intrinsics}

        depth = self.sess.run(self.outputs['depth_reprojection'], feed_dict=feed_dict)
        return fill_depth(depth)

    def deepv2d_init(self):
        if self.use_fcrn:
            if self.mode == 'keyframe':
                feed_dict = {self.images_placeholder: self.images[[0]]}
            else:
                feed_dict = {self.images_placeholder: self.images}
            
            self.depths = self.sess.run(self.outputs['fcrn'], feed_dict=feed_dict)

        else:
            if self.mode == 'keyframe':
                images = np.stack([self.images[0]] * self.images.shape[0], axis=0) # 升级维度复制
                poses = np.stack([np.eye(4)] * self.images.shape[0], axis=0) #复制8次单位矩阵
                # 在这里进行初始赋值
                feed_dict = {
                    self.images_placeholder: images,
                    self.poses_placeholder: poses,
                    self.intrinsics_placeholder: self.intrinsics}

            else:
                ii = np.arange(self.images.shape[0])
                adj = np.stack([ii, ii], axis=-1)

                feed_dict = {
                    self.images_placeholder: self.images,
                    self.poses_placeholder: self.poses,
                    self.adj_placeholder: adj,
                    self.intrinsics_placeholder: self.intrinsics}

            self.depths = self.sess.run(self.outputs['depths'], feed_dict=feed_dict)

    def update_poses(self, itr=0):
        n = self.images.shape[0]
        # 生成矩阵网格点坐标
        if self.mode == 'keyframe':
            ii, jj = np.meshgrid(np.arange(1), np.arange(1,n))
        else:
            ii, jj = np.meshgrid(np.arange(n), np.arange(n))
        
        ii = ii.reshape(-1) # 转变为一维度向量
        jj = jj.reshape(-1) # 1-8数组
        v = ~np.equal(ii, jj) # 1*7的True数组

        # don't use pairs with self loop
        edges = np.stack([ii[v], jj[v]], axis=-1) #7*2的数组，[0,1],[0,2]...

        feed_dict = {
            self.images_placeholder: self.images,
            self.depths_placeholder: self.depths,
            self.poses_placeholder: self.poses,
            self.edges_placeholder: edges,
            self.init_placeholder: (itr==0),
            self.intrinsics_placeholder: self.intrinsics}
            
        # execute pose subgraph，主要获取，姿势信息和相机参数
        outputs = [self.outputs['poses'], self.outputs['intrinsics'], self.outputs['weights']]
        # 在这里进行推理和训练
        #self.poses, self.intrinsics, self.weights = self.sess.run(outputs, feed_dict=feed_dict) 
        self.poses,  intrinsics,  weights = self.sess.run(outputs, feed_dict=feed_dict) 

        if not self.cfg.MOTION.IS_CALIBRATED:
            print("intrinsics (fx, fy, cx, cy): ", self.intrinsics)

    def update_depths(self, itr=0):
        """[summary]

        Args:
            itr (int, optional): [description]. Defaults to 0.
        """
        # 获取图像长度
        n = self.images.shape[0]
        inds_list = []

        if self.mode == 'keyframe':
            feed_dict = {
                self.images_placeholder: self.images,
                self.poses_placeholder: self.poses,
                self.intrinsics_placeholder: self.intrinsics}
        
            self.depths = self.sess.run(self.outputs['depths'], feed_dict=feed_dict)

        else:
            # 进行迭代
            for i in range(n):
                inds = np.arange(n).tolist()
                inds.remove(i)
                inds = [i] + inds
                inds_list.append(inds)

            adj_list = np.array(inds_list, dtype=np.int32)

            if n <= 4:
                feed_dict = {
                    self.images_placeholder: self.images,
                    self.poses_placeholder: self.poses,
                    self.adj_placeholder: adj_list,
                    self.intrinsics_placeholder: self.intrinsics}

                self.depths = self.sess.run(self.outputs['depths'], feed_dict=feed_dict)

            else: # we need to split up inference to fit in memory
                s = 2
                for i in range(0, n, s):
                    feed_dict = {
                        self.images_placeholder: self.images,
                        self.poses_placeholder: self.poses,
                        self.adj_placeholder: adj_list[i:i+s],
                        self.intrinsics_placeholder: self.intrinsics}

                    self.depths[i:i+s] = self.sess.run(self.outputs['depths'], feed_dict=feed_dict)


    def vizualize_output(self, inds=[0]):
        feed_dict = {
            self.images_placeholder: self.images,
            self.depths_placeholder: self.depths,
            self.poses_placeholder: self.poses,
            self.intrinsics_placeholder: self.intrinsics}

        keyframe_image = self.images[0]
        keyframe_depth = self.depths[0]

        image_depth = vis.create_image_depth_figure(keyframe_image, keyframe_depth)
        cv2.imwrite('depth.png', image_depth)
        #cv2.imshow('image_depth', image_depth/255.0)
        
        # print("Press any key to cotinue")
        #cv2.waitKey()

        # use depth map to create point cloud
        point_cloud, point_colors = self.sess.run(self.outputs['point_cloud'], feed_dict=feed_dict)

        print("Press q to exit")
        vis.visualize_prediction(point_cloud, point_colors, self.poses)

    def inference(self, images, poses,intrinsics=None, iters=2, viz=False):
        self.images = images
        self.poses = poses
        self.intrinsics = intrinsics
        # 进行数据绑定
        feed_dict = {
                self.images_placeholder: self.images,
                self.poses_placeholder: self.poses,
                self.intrinsics_placeholder: self.intrinsics
            }
        # 开始执行推理
        self.depths = self.sess.run(self.outputs['depths'], feed_dict=feed_dict)

        return self.depths
    # call基本函数,用来执行推理
    def __call__(self, images, intrinsics=None, iters=2, viz=False):
        n_frames = len(images) # 8张图像
        self.images = np.stack(images, axis=0) # 将所有图片进行维度叠加

        if intrinsics is None:
            # initialize intrinsics
            fx = images.shape[2] * 1.2
            fy = images.shape[2] * 1.2
            cx = images.shape[2] / 2.0
            cy = images.shape[1] / 2.0
            intrinsics = np.stack([fx, fy, cx, cy])


        # (fx, fy, cx, cy) # 初步估计相机内参
        self.intrinsics = intrinsics # 获取初始值
        poses = np.eye(4).reshape(1, 4, 4) # 1个单位矩阵4*4
        poses = np.tile(poses, [n_frames, 1, 1]) # 给每个frame一个相机参数矩阵，有8*4*4的矩阵
        self.poses = poses #初始化位姿

        # initalize reconstruction
        self.deepv2d_init() 

        for i in range(iters):
            print("start iterator {}".format(i))
            time_start=time.time()
            #self.update_poses(i)    # 计算位姿
            self.update_depths()    # 计算深度
            time_end=time.time()
            print('time cost',time_end-time_start,'ms')
        if viz:
            self.vizualize_output()

        return self.depths, self.poses

