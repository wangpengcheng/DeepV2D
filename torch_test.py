import sys
import os
sys.path.append('deepv2d')
sys.path.append('evaluation')
import torch
import time
import cv2
import matplotlib.pyplot as plt
from deepv2d.modules.depth_module import DepthModule
from data_stream.tum import TUM_RGBD
from core import config
from deepv2d import vis
import eval_utils 


# 加载数据集进行推理

def inference_test(deepModel, cfg):
    deepModel = deepModel.load_state_dict(torch.load('pytorch/tum/tmu_model/depth.pth'))
    deepModel.cuda()

    db = TUM_RGBD(cfg.INPUT.RESIZE, "data/tum", r=4)

    trainloader = torch.utils.data.DataLoader(db, batch_size=1, shuffle=False, num_workers=8)
    time_sum =0.0
    iter_number = len(db)
    for i, data in enumerate(trainloader, 0):
        images_batch, poses_batch, gt_batch, filled_batch, pred_batch, intrinsics_batch, frame_id= data
                    #images_batch, gt_batch, intrinsics_batch =  prefetcher.next()
                    # 进行数据预处理,主要是维度交换
        images = images_batch.permute(0, 1, 4, 2, 3)
        Ts = poses_batch.cuda()
        images = images.cuda()
        intrinsics_batch = intrinsics_batch.cuda()
        gt_batch = gt_batch.cuda()
        # 计算时间
        time_start=time.time()
        outputs = deepModel(Ts, images, intrinsics_batch)
        time_end=time.time()
        key_frame_depth = outputs[0]
        # 关键rgb帧
        key_frame_image = images_batch[0][0]
        # 关键深度帧
        depth_gt = gt_batch[0]
        # 计算深度缩放
        scalor = eval_utils.compute_scaling_factor(depth_gt.cpu().detach().numpy(), key_frame_depth.cpu().detach().numpy(), min_depth=0.1, max_depth=10.0)
        key_frame_depth =  scalor * key_frame_depth.cpu().detach().numpy()
        # 对深度图像进行平滑处理
        # key_frame_depth = cv2.medianBlur(key_frame_depth,5)
        image_depth = vis.create_image_depth_figure(key_frame_image.cpu().detach().numpy(), key_frame_depth)
        result_out_dir = "{}/{}".format("data/tum", "inference_result")
        # 检测路径文件夹
        if not os.path.exists(result_out_dir):
            os.makedirs(result_out_dir)
        cv2.imwrite("{}/{}.png".format(result_out_dir, i), image_depth)
        print("wirte image:{}/{}.png".format(result_out_dir,i))
        if i != 0:
            time_sum = time_sum + (time_end-time_start)
        print('time cost',time_end-time_start,'s')
    print("{} images,totle time: {} s, avg time: {} s".format(iter_number-1, time_sum, time_sum/(iter_number-1)))



if __name__ == '__main__':
    cfg = config.cfg_from_file("cfgs/tum_2_2_fast_shufflev2.yaml")
    deepModel = DepthModule(cfg)
    inference_test(deepModel,cfg)

# print(model)
# model.load_state_dict(torch.load('pytorch/tum/tmu_model/depth.pth'))


# for i in model.state_dict():
#     print(i)