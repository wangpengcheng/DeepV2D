import numpy as np
import csv
import os
import time
import random
import glob
import os.path as osp


from geometry.transformation import *
import cv2
import tensorflow as tf


fx = 5.1885790117450188e+02
fy = 5.1946961112127485e+02
cx = 3.2558244941119034e+02
cy = 2.5373616633400465e+02
# 相机基本参数
intrinsics = np.array([fx, fy, cx, cy], dtype=np.float32)


from scipy import interpolate


def fill_depth(depth):
    # 创建深度网络图
    x, y = np.meshgrid(np.arange(depth.shape[1]).astype("float32"),
                       np.arange(depth.shape[0]).astype("float32"))
    xx = x[depth > 0] # 只留大于0的点
    yy = y[depth > 0] #
    
    zz = depth[depth > 0]
    # 进行二维插值
    grid = interpolate.griddata((xx, yy), zz.ravel(),
                                (x, y), method='nearest')
    # 网格数据
    return grid
# 将四元数组转换为旋转矩阵
def quat2rotm(q):
    """Convert quaternion into rotation matrix """
    q /= np.sqrt(np.sum(q**2)) 
    x, y, z, s = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r1 = np.stack([1-2*(y**2+z**2), 2*(x*y-s*z), 2*(x*z+s*y)], axis=1)
    r2 = np.stack([2*(x*y+s*z), 1-2*(x**2+z**2), 2*(y*z-s*x)], axis=1)
    r3 = np.stack([2*(x*z-s*y), 2*(y*z+s*x), 1-2*(x**2+y**2)], axis=1)
    return np.stack([r1, r2, r3], axis=1)

# 将位姿转换为矩阵
def pose_vec2mat(pvec, use_filler=True):
    """Convert quaternion vector represention to SE3 group"""
    t, q = pvec[np.newaxis, 0:3], pvec[np.newaxis, 3:7]
    R = quat2rotm(q)
    t = np.expand_dims(t, axis=-1)
    # 最终的转换矩阵
    P = np.concatenate([R, t], axis=2)
    if use_filler:
        filler = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 1, 4])
        P = np.concatenate([P, filler], axis=1)
    return P[0]

    
class NYUv2:
    def __init__(self, dataset_path, mode='train', use_filled=False, n_frames=4, max_dt=1.25):
        self.n_frames = n_frames
        self.max_dt = max_dt
        self.height = 480
        self.width = 640
        
        self.dataset_path = dataset_path
        self.mode = mode
        self.use_filled = use_filled
        # 构建数据索引
        self.build_dataset_index()

    def copy(self):
        db = NYUv2(dataset_path=self.dataset_path, 
                   mode=self.mode, 
                   use_filled=self.use_filled,
                   n_frames=self.n_frames,
                   max_dt=self.max_dt)

        random.shuffle(db.dataset_index)
        return db

    def get_dims(self):
        return (self.n_frames, self.height, self.width)

    def shape(self):
        return (self.n_frames, self.height, self.width)

    def __len__(self):
        return len(self.dataset_index)

    def __getitem__(self, index):
        data_blob = self.dataset_index[index]
        # 获取帧数
        num_frames = data_blob['n_frames']
        # 序列帧数
        num_samples = self.n_frames
        # 随机索引坐标
        inds = np.random.choice(num_frames, num_samples, replace=False)
        # 选取第一个为关键帧
        keyframe_index = inds[0]

        images = []
        for i in inds:
            image_file = data_blob['images'][i]
            images.append(cv2.imread(image_file))
        # 深度数据文件
        depth_file = data_blob['depths'][keyframe_index]
        # 进行图像读取
        depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
        # 进行深度读取
        depth = (depth.astype(np.float32)) / 5000.0
        # 深度平滑处理
        filled = fill_depth(depth)
        # 关键帧序列
        frameid = data_blob['ids'][keyframe_index]
        # 转换为
        frameid = np.int32(frameid)
    
        poses = []
        for i in inds:
            pose_vec = data_blob['poses'][i]
            pose_mat = pose_vec2mat(pose_vec)
            poses.append(np.linalg.inv(pose_mat))

        images = np.stack(images, axis=0).astype(np.uint8)
        poses = np.stack(poses, axis=0).astype(np.float32)

        kvec = intrinsics.copy()
        return images, poses, depth, filled, filled, kvec, frameid

    def __iter__(self):
        random.shuffle(self.dataset_index)
        while 1:
            self.perm = np.arange(len(self.dataset_index))
            np.random.shuffle(self.perm)
            for i in self.perm:
                yield self.__getitem__(i)

    def build_dataset_index(self):
        # 获取所有的深度学习队列
        self.all_image_depth_pairs = {}
        self.frame_id_mapping = {}
        self.dataset_index = []
        self.frameid = 0
        # 读取场景列表
        scenes_list = 'data/nyu/train_scenes.txt'
        # 获取所有场景
        reader = csv.reader(open(scenes_list))
        train_scenes = [x[0] for x in reader]
        # 获取全部场景
        dataset_scenes = []
        for sceneb in train_scenes:
            seach_str = os.path.join(self.dataset_path, '%s*'%sceneb)
            
            for scene in glob.glob(seach_str):
                dataset_scenes.append(os.path.basename(scene))
        # 遍历每一个场景
        for scene in dataset_scenes:
            # 场景文件夹
            scene_dir = os.path.join(self.dataset_path, scene)
            # 获取时间戳
            timestamps, pairs, poses = self._load_trajectory(scene_dir)
            # 通过时间戳获取训练样例
            indicies = self._gather_training_examples_from_timestamps(timestamps, max_dt=self.max_dt)

            for inds in indicies:
                training_example = {
                    "n_frames": len(inds),
                    "intrinsics": intrinsics,
                }
                training_example["images"] = []
                training_example["depths"] = []
                training_example["filled"] = []
                training_example["poses"] = []
                training_example["ids"] = []

                for i in inds:
                    image_file = osp.join(self.dataset_path, scene, pairs[i][0])
                    depth_file = osp.join(self.dataset_path, scene, pairs[i][1])
                    filld_file = osp.join(self.dataset_path, scene, pairs[i][1].replace("depth", "filled"))
                    if osp.isfile(filld_file) and self.use_filled:
                       training_example["filled"].append(filld_file)
                    else:
                        training_example["filled"].append(depth_file)
                    training_example['images'].append(image_file)
                    training_example['depths'].append(depth_file)
                    training_example['poses'].append(poses[i])

                    if image_file not in self.frame_id_mapping:
                        self.frame_id_mapping[image_file] = self.frameid
                        self.frameid += 1

                    self.all_image_depth_pairs[image_file] = depth_file
                    training_example['ids'].append(self.frame_id_mapping[image_file])

                self.dataset_index.append(training_example)
        
    def _load_trajectory(self, scene_dir):
        # 对应文件
        associations_file = osp.join(scene_dir, 'associations.txt')
        # 相机位姿
        camera_file = osp.join(scene_dir, 'pose.txt')
        # 检查是否存在
        if not (osp.isfile(associations_file) and osp.isfile(camera_file)):
            return [], [], []

        pairs_dict = {}
        with open(associations_file) as f:
            # 进行文件读取
            reader = csv.reader(f, delimiter=' ')
            # 根据时间戳构造字典
            for row in reader:
                pairs_dict[row[0]] = (row[1], row[3])

        poses = []
        # 最终合成效果
        pairs = []
        timestamps = []
        # 打开文件位姿信息
        with open(camera_file) as f:
            # 读取文件
            reader = csv.reader(f, delimiter=' ')
            # 进行遍历
            for row in reader:
                # 获取时间戳和向量矩阵
                tstamp, vec = row[0], [float(x) for x in row[1:]]
                # 传唤类型
                vec = np.array(vec).astype(np.float32)
                # 添加pose
                poses.append(vec)
                # 将关键对添加
                pairs.append(pairs_dict[tstamp])
                # 构造时间戳
                timestamps.append(float(tstamp))
        # 时间戳、深度和图像、位姿信息
        return timestamps, pairs, poses
    
    def _gather_training_examples_from_timestamps(self, timestamps, dt=0.1, max_dt=1.25):

        indicies = []
        if len(timestamps) == 0:
            return indicies
        # 时间戳开始时间
        t0 = timestamps[0]
        # 关键帧坐标
        keyframe_inds = [0]
        # 帧时间
        keyframe_ts = [t0]

        i = 1
        while i<len(timestamps):
            # 获取时间戳
            t1 = timestamps[i]
            # 进行比较
            if t1-t0 > dt:
                # 添加索引
                keyframe_inds.append(i)
                # 添加时间戳
                keyframe_ts.append(t1)
                # 更新时间戳
                t0 = t1
            i = i + 1
        # 时间戳队列
        ts = np.array(keyframe_ts, dtype=np.float64)
        # 关键帧队列
        ki = np.array(keyframe_inds, dtype=np.int32)
        # 筛选出关键帧
        for i, t in zip(keyframe_inds, keyframe_ts):
            # 进行索引比较
            inds, = np.where(np.abs(t-ts) < max_dt)
            # 将其添加到数组中
            if len(inds) > self.n_frames:
                indicies.append(ki[inds].tolist())

        return indicies

    def load_scene(self, scene):
        """
        加载场景文件夹

        Args:
            scene ([type]): [description]

        Returns:
            [type]: [description]
        """
        image_list = os.path.join(self.dataset_path, scene, 'rgb.txt')
        timestamps, images = [], []
        with open(image_list) as f:
            reader = csv.reader(f, delimiter=' ')
            for row in reader:
                timestamps.append(row[0])
                images.append(row[1])
        return timestamps, images

    def test_iterator(self, r=2, dt=0.5):
        # 加载文件夹
        test_frames = np.loadtxt('nyu_test_files.txt', dtype=np.unicode_)
        for test_frame in test_frames:
            scene, image_id = test_frame.split('/')
            query = os.path.join('rgb', image_id.split('-')[1] + '.png')

            timestamps, imfiles = self.load_scene(scene)
            tstamps = np.array(timestamps, dtype=np.float64)

            i = imfiles.index(query)
            t = tstamps[i]

            tquery = t + dt*np.concatenate((np.arange(-r, 0), np.arange(1, r+1)))
            ixs = np.argmin(np.abs(tstamps[:, np.newaxis] - tquery), axis=0)

            image_files = [os.path.join(self.dataset_path, scene, query)]
            for ix in ixs:
                image_files.append(os.path.join(self.dataset_path, scene, imfiles[ix]))

            images = []
            for image_file in image_files:
                image = cv2.imread(image_file)
                images.append(image)

            images = np.stack(images, axis=0).astype(np.uint8)
            yield images, intrinsics.copy()


    def iterate_sequence(self, seq):
        sequence_path = os.path.join(self.dataset_path, seq)
        image_files = sorted(glob.glob(os.path.join(sequence_path, 'rgb', '*.png')))

        for image_file in image_files[::4]:
            image = cv2.imread(image_file)
            yield image, intrinsics.copy()




