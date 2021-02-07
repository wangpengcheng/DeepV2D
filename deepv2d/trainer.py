import tensorflow as tf
import numpy as np
import os
import cv2

from data_layer import DataLayer, DBDataLayer

from geometry.transformation import *
from utils.memory_saving_gradients import gradients
from utils.average_grads import average_gradients
from utils import mem_util

from modules.depth import DepthNetwork
from modules.motion import MotionNetwork

gpu_no = '0' # or '1'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_no

MOTION_LR_FRACTION = 0.1

class DeepV2DTrainer(object):
    """
    网络训练基础类型类
    Args:
        object ([type]): [description]
    """
    def __init__(self, cfg):
        self.cfg = cfg
    # 第一阶段训练，主要是深度估计网络的训练
    def build_train_graph_stage1(self, cfg, num_gpus=1):
        # 读取基本参数
        id_batch, images_batch, poses_batch, gt_batch, filled_batch, pred_batch, intrinsics_batch = self.dl.next()
        # 对参数进行分割
        images_batch = tf.split(images_batch, num_gpus)
        poses_batch = tf.split(poses_batch, num_gpus)
        gt_batch = tf.split(gt_batch, num_gpus)
        filled_batch = tf.split(filled_batch, num_gpus)
        pred_batch = tf.split(pred_batch, num_gpus)
        intrinsics_batch = tf.split(intrinsics_batch, num_gpus)
        # 执行调度器
        with tf.name_scope("training_schedule"):
            global_step = tf.Variable(0, name='global_step', trainable=False)
            lr = tf.train.exponential_decay(cfg.TRAIN.LR, global_step, 5000, 0.5, staircase=True)
            optim = tf.train.RMSPropOptimizer(MOTION_LR_FRACTION * lr)

        tower_grads = []
        tower_losses = []
        # 枚举GPU
        for gpu_id in range(num_gpus):
            images = images_batch[gpu_id]
            poses = poses_batch[gpu_id]
            depth_gt = gt_batch[gpu_id]
            depth_filled = filled_batch[gpu_id]
            depth_pred = pred_batch[gpu_id]
            intrinsics = intrinsics_batch[gpu_id]

            Gs = VideoSE3Transformation(matrix=poses)
            # 创建位姿估计网络
            motion_net = MotionNetwork(cfg.MOTION, bn_is_training=True, reuse=gpu_id>0)

            with tf.device('/gpu:%d' % gpu_id):
                # 获取深度信息
                depth_input = tf.expand_dims(depth_filled, 1)
                # 前向计算
                Ts, kvec = motion_net.forward(None, images, depth_input, intrinsics)
                # 计算loss值
                total_loss = motion_net.compute_loss(Gs, depth_input, intrinsics, log_error=(gpu_id==0))
                # 计算总loss
                tower_losses.append(total_loss)
                # 显示数据
                var_list = tf.trainable_variables()
                # 计算梯度下降
                grads = gradients(total_loss, var_list)
                # 定义计算出来的梯度函数
                gvs = []
                for (g, v) in zip(grads, var_list):
                    if g is not None:
                        if cfg.TRAIN.CLIP_GRADS:
                            g = tf.clip_by_value(g, -1.0, 1.0)
                        gvs.append((g,v))

                gvs = zip(grads, var_list)
                tower_grads.append(gvs)

                # use last gpu to compute batch norm statistics
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)


        with tf.name_scope("train_op"):
            # 计算平均梯度
            gvs = average_gradients(tower_grads)
            # 求平均均值
            total_loss = tf.reduce_mean(tf.stack(tower_losses, axis=0))

            with tf.control_dependencies(update_ops):
                self.train_op = optim.apply_gradients(gvs, global_step)

            self.write_op = None
            self.total_loss = total_loss
            tf.summary.scalar("learning_rate", lr)
            tf.summary.scalar("total_loss", total_loss)

    # 构建二段训练，主要是相机位姿和深度信息的整合
    def build_train_graph_stage2(self, cfg, num_gpus=1):

        with tf.name_scope("training_schedule"):
            # 全局训练参数
            global_step = tf.Variable(0, name='global_step', trainable=False)
            # 将其转换为folat
            gs = tf.to_float(global_step)
            # 
            if cfg.TRAIN.RENORM:
                rmax = tf.clip_by_value(5.0*(gs/2.5e4)+1.0, 1.0, 5.0) # rmax schedule
                dmax = tf.clip_by_value(8.0*(gs/2.5e4), 0.0, 8.0) # dmax schedule
                rmin = 1.0 / rmax
                schedule = {'rmax': rmax, 'rmin': rmin, 'dmax': dmax}
            else:
                schedule = None

            LR_DECAY = int(0.8 * self.training_steps)
            lr = tf.train.exponential_decay(cfg.TRAIN.LR, global_step, LR_DECAY, 0.2, staircase=True)

            stereo_optim = tf.train.RMSPropOptimizer(lr)
            motion_optim = tf.train.RMSPropOptimizer(MOTION_LR_FRACTION*lr)
        # 进行数据读取
        id_batch, images_batch, poses_batch, gt_batch, filled_batch, pred_batch, intrinsics_batch = self.dl.next()
        print("image shape:{}".format(images_batch))
        # 在这里进行数据分割
        images_batch = tf.split(images_batch, num_gpus)
        poses_batch = tf.split(poses_batch, num_gpus)
        gt_batch = tf.split(gt_batch, num_gpus)
        filled_batch = tf.split(filled_batch, num_gpus)
        pred_batch = tf.split(pred_batch, num_gpus)
        intrinsics_batch = tf.split(intrinsics_batch, num_gpus)
        # 计算梯度
        tower_motion_grads = []
        tower_stereo_grads = []
        tower_predictions = []
        tower_losses = []
        write_ops = []

        for gpu_id in range(num_gpus):
            if cfg.MOTION.USE_MOTION:
                # 位姿估计网络
                motion_net = MotionNetwork(cfg.MOTION, reuse=gpu_id>0)
            else:
                print("Donot Use motion net")
            # 深度估计网络
            depth_net = DepthNetwork(cfg.STRUCTURE, schedule=schedule, reuse=gpu_id>0)
            # 数据分配
            # 图像
            images = images_batch[gpu_id]
            # 位姿
            poses = poses_batch[gpu_id]
            # 深度差值
            depth_gt = gt_batch[gpu_id]
            # 深度填充
            depth_filled = filled_batch[gpu_id]
            # 获取预测值
            depth_pred = pred_batch[gpu_id]
            # 相机深度
            intrinsics = intrinsics_batch[gpu_id]
            # 将位姿转换为SE3,对应的空间变换矩阵
            Gs = VideoSE3Transformation(matrix=poses)
            # 获取相关数据
            batch, frames, height, width, _ = images.get_shape().as_list()
            # 使用命名空间
            with tf.name_scope("depth_input"):
                # 定义学习率和指数衰减 https://blog.csdn.net/lllxxq141592654/article/details/84110600
                input_prob = tf.train.exponential_decay(2.0, global_step, LR_DECAY, 0.02, staircase=False)
                rnd = tf.random_uniform([], 0, 1)
                depth_input = tf.cond(rnd<input_prob, lambda: depth_filled, lambda: depth_pred)

            with tf.device('/gpu:%d' % gpu_id):
                if cfg.MOTION.USE_MOTION:
                    # 位姿估计网络
                    # 前向推理获取位姿矩阵，和相机参数
                    # motion inference
                    Ts, kvec = motion_net.forward(None, images, depth_input[:,tf.newaxis], intrinsics)
                else:
                    # 直接使用输入的相机位姿和参数
                    Ts = Gs
                    kvec = intrinsics
                
                # 在这里直接将位姿矩阵和相机参数进行初始化

                # 是否停止迭代
                stop_cond = global_step < cfg.TRAIN.GT_POSE_ITERS
                # 转换后的坐标
                Ts = cond_transform(stop_cond, Ts.copy(stop_gradients=True), Ts)
                # 最终的相机参数
                kvec = tf.cond(stop_cond, lambda: tf.stop_gradient(kvec), lambda: kvec)
                # depth inference
                # 进行前向计算推理，获取深度预测值
                depth_pr = depth_net.forward(Ts, images, kvec)
                # 计算
                depth_loss = depth_net.compute_loss(depth_gt, log_error=(gpu_id==0))
                if cfg.MOTION.USE_MOTION:
                    # 计算loss值
                    motion_loss = motion_net.compute_loss(Gs,
                        depth_filled[:,tf.newaxis], intrinsics, log_error=(gpu_id==0))


                # compute all gradients
                if 1:
                    if cfg.MOTION.USE_MOTION:
                        total_loss = cfg.TRAIN.DEPTH_WEIGHT * depth_loss + motion_loss
                    else:
                        total_loss = cfg.TRAIN.DEPTH_WEIGHT * depth_loss
            
                    var_list = tf.trainable_variables()
                    # 计算所有梯度
                    grads = gradients(total_loss, var_list)

                # split backward pass
                else:
                    so3, translation = Ts.so3, Ts.translation
                    stereo_vars = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope="stereo")
                    stereo_grads = gradients(depth_loss, [so3, translation] + stereo_vars)
                    diff_so3, diff_translation, stereo_grads = \
                        stereo_grads[0], stereo_grads[1], stereo_grads[2:]
                    
                    if cfg.MOTION.USE_MOTION:
                        motion_vars = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope="motion")
                        motion_grads = tf.gradients([motion_loss, so3, translation], motion_vars, 
                        grad_ys=[tf.ones_like(motion_loss), diff_so3, diff_translation])
                    

                    if cfg.MOTION.USE_MOTION:
                        grads = stereo_grads + motion_grads
                        var_list = stereo_vars + motion_vars
                    else:
                        grads = stereo_grads
                        var_list = stereo_vars

                motion_gvs = []
                stereo_gvs = []
                for (g, v) in zip(grads, var_list):
                        if 'stereo' in v.name and (g is not None):
                            if cfg.TRAIN.CLIP_GRADS:
                                g = tf.clip_by_value(g, -1.0, 1.0)
                            stereo_gvs.append((g,v))

                        if 'motion' in v.name and (g is not None):
                            if cfg.TRAIN.CLIP_GRADS and (g is not None):
                                g = tf.clip_by_value(g, -1.0, 1.0)
                            motion_gvs.append((g,v))

                tower_stereo_grads.append(stereo_gvs)
                if cfg.MOTION.USE_MOTION:
                    tower_motion_grads.append(motion_gvs)
                
                tower_predictions.append(depth_pr)
                tower_losses.append(depth_loss)

                if gpu_id == 0:
                    self.total_loss = depth_loss

                # use last gpu to compute batch norm statistics
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        tower_stereo_gvs = average_gradients(tower_stereo_grads)
        # 计算平均梯度
        if cfg.MOTION.USE_MOTION:
            tower_motion_gvs = average_gradients(tower_motion_grads)
        
        with tf.name_scope("train_op"):
            with tf.control_dependencies(update_ops):
                if cfg.MOTION.USE_MOTION:
                    self.train_op = tf.group(
                        stereo_optim.apply_gradients(tower_stereo_gvs),
                        motion_optim.apply_gradients(tower_motion_gvs),
                        tf.assign(global_step, global_step+1)
                    )
                else:
                    self.train_op = tf.group(
                        stereo_optim.apply_gradients(tower_stereo_gvs),
                        tf.assign(global_step, global_step+1)
                    )

        self.write_op = self.dl.write(id_batch, tf.concat(tower_predictions, axis=0))
        self.total_loss = tf.reduce_mean(tf.stack(tower_losses, axis=0))

        tf.summary.scalar("total_loss", self.total_loss)
        tf.summary.scalar("learning_rate", lr)
        tf.summary.scalar("input_prob", input_prob)


    def train(self, data_source, cfg, stage=1, ckpt=None, restore_ckpt=None, num_gpus=1):
        """主要的训练函数

        Args:
            data_source ([type]): [description]
            cfg ([type]): [description]
            stage (int, optional): [description]. Defaults to 1.
            ckpt ([type], optional): [description]. Defaults to None.
            restore_ckpt ([type], optional): [description]. Defaults to None.
            num_gpus (int, optional): [description]. Defaults to 1.

        Returns:
            [type]: [description]
        """
        batch_size = num_gpus * cfg.TRAIN.BATCH[stage-1]
        max_steps = cfg.TRAIN.ITERS[stage-1]
        self.training_steps = max_steps
        # 开始加载数据模型
        print ("batch size: %d \t max steps: %d"%(batch_size, max_steps))
        # 开始加载数据模型
        if isinstance(data_source, str):
            self.dl = DataLayer(data_source, batch_size=batch_size)
        else:
            self.dl = DBDataLayer(data_source, batch_size=batch_size)
        # 第一阶段
        if stage == 1:
            self.build_train_graph_stage1(cfg, num_gpus=num_gpus)
        # 第二阶段
        elif stage == 2:
            self.build_train_graph_stage2(cfg, num_gpus=num_gpus)

        # 进行数据合并
        self.summary_op = tf.summary.merge_all()
        # 进行模型存储
        saver = tf.train.Saver([var for var in tf.model_variables()], max_to_keep=10)
        # 写入日志信息
        train_writer = tf.summary.FileWriter(cfg.LOG_DIR+'_stage_%s'%str(stage)) # 写入到数据训练文件中
        # 进行初始化
        init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())
        # 设置存储频率
        SUMMARY_FREQ = 10
        # 设置日志频率
        LOG_FREQ = 100
        # 设置checkpoint中间输出频率
        CHECKPOINT_FREQ = 5000
        # 定义TensorFlow配置
        config = tf.ConfigProto()
        # 配置GPU内存分配方式，按需增长，很关键
        config.gpu_options.allow_growth = True

        # 配置可使用的显存比例，为所有显存的80%
        config.gpu_options.per_process_gpu_memory_fraction = 0.9

        # 在创建session的时候把config作为参数传进去
        sess = tf.InteractiveSession(config = config)

        with tf.Session() as sess:
            sess.run(init_op)

            # train with tfrecords 
            # 数据加载层
            if isinstance(self.dl, DataLayer):
                coord = tf.train.Coordinator() # 创建线程协调器
                threads = tf.train.start_queue_runners(coord=coord) # 创建任务队列
 
            # train with python data loader
            elif isinstance(self.dl, DBDataLayer):
                self.dl.init(sess)

            kwargs = {}
            # 训练阶段大于2，两次训练都结束,保存姿态网络数据
            if stage >= 2 and cfg.MOTION.USE_MOTION:
                # 位姿估计的所有变量
                motion_vars = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope="motion")
                # 创建存储
                motion_saver = tf.train.Saver(motion_vars)
                # 进行中间参数保存，保存的模型
                if ckpt is not None:
                    motion_saver.restore(sess, ckpt)
                # 存储的临时文件
                if restore_ckpt is not None:
                    saver.restore(sess, restore_ckpt)
            
            # 运行时的loss
            running_loss = 0.0
            # 开始迭代并进行训练
            for step in range(1, max_steps):

                kwargs = {}
                fetches = {}
                fetches['train_op'] = self.train_op
                fetches['loss'] = self.total_loss
                if self.write_op is not None:
                    fetches['write_op'] = self.write_op
                # 进行关键信息存储
                if step % SUMMARY_FREQ == 0:
                    fetches['summary'] = self.summary_op
                # 执行命令,进行运算
                result = sess.run(fetches, **kwargs)

                if step % SUMMARY_FREQ == 0:
                    train_writer.add_summary(result['summary'], step)

                if step % LOG_FREQ == 0:
                    print('[stage=%d, %5d] loss: %.9f'%(stage, step, running_loss / LOG_FREQ))
                    running_loss = 0.0
                # 存储模型文件
                if step % CHECKPOINT_FREQ == 0:
                    checkpoint_file = os.path.join(cfg.CHECKPOINT_DIR, '_stage_%s.ckpt'%str(stage))
                    saver.save(sess, checkpoint_file, step)
                # 进行loss计算
                running_loss += result['loss']
            # 存储最终模型文件
            checkpoint_file = os.path.join(cfg.CHECKPOINT_DIR, '_stage_%s.ckpt'%str(stage))
            # 存储模型文件
            saver.save(sess, checkpoint_file)

        return checkpoint_file
