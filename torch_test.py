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
from utils.my_utils import *
#from torch2trt import torch2trt

# 加载数据集进行推理

def inference_test(deepModel, cfg):
   
    #deepModel = deepModel.load_state_dict(torch.load('pytorch/tum/tmu_model/depth.pth'))
    
    db = TUM_RGBD(cfg.INPUT.RESIZE, "data/mydata2",test=False, r=2)

    trainloader = torch.utils.data.DataLoader(db, batch_size=1, shuffle=False, num_workers=8)
    time_sum =0.0
    iter_number = len(db)
    deepModel.cuda()
    with torch.no_grad():
        deepModel.eval()
        for i, data in enumerate(trainloader, 0):
            images_batch, poses_batch, gt_batch, filled_batch, pred_batch, intrinsics_batch, frame_id= data
                        # 进行数据预处理,主要是维度交换
            images = images_batch.permute(0, 1, 4, 2, 3)
            images, gt_batch, intrinsics_batch, a = prepare_inputs(cfg , images, gt_batch, intrinsics_batch)
            Ts = poses_batch.cuda()
            images = images.float().cuda()
            intrinsics_batch = intrinsics_batch.float().cuda()
            gt_batch = gt_batch.cuda()
            # print(images.shape)
            # print(Ts.shape)
            # print(intrinsics_batch.shape)
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
            image_depth = vis.create_ex_image_depth_figure(key_frame_image.cpu().detach().numpy(), depth_gt.cpu().detach().numpy(), key_frame_depth)
            result_out_dir = "{}/{}".format("data/mydata_test", "inference_result_1")
            # 检测路径文件夹
            if not os.path.exists(result_out_dir):
                os.makedirs(result_out_dir)
            cv2.imwrite("{}/{}.png".format(result_out_dir, i), image_depth)
            print("wirte image:{}/{}.png".format(result_out_dir,i))
            if i != 0:
                time_sum = time_sum + (time_end-time_start)
            print('time cost',time_end-time_start,'s')
        print("{} images,totle time: {} s, avg time: {} s".format(iter_number-1, time_sum, time_sum/(iter_number-1)))

def converToTensorrt(deepModel, cfg):
    model = deepModel.eval().cuda()
    db = TUM_RGBD(cfg.INPUT.RESIZE, "data/tum2", r=2)

    trainloader = torch.utils.data.DataLoader(db, batch_size=1, shuffle=False, num_workers=8)
    for i, data in enumerate(trainloader, 0):
        if i> 0:
            break
        images_batch, poses_batch, gt_batch, filled_batch, pred_batch, intrinsics_batch, frame_id= data
        images = images_batch.permute(0, 1, 4, 2, 3)
        poses = poses_batch.cuda()
        images = images.float().cuda()
        intrinsics = intrinsics_batch.float().cuda()
        with torch.no_grad():
            print("hello")
            outputs = deepModel(poses, images, intrinsics)
            #print(outputs)
            model_trt = torch2trt(model, [poses, images, intrinsics ])
        #torch.save(model_trt.state_dict(), 'deep_trt.pth')
    # images = torch.ones(1, 5, 3, 240, 320).float().cuda()
    # poses = torch.ones(1, 5, 4, 4).float().cuda()
    # intrinsics = torch.ones(1, 4).float().cuda()

def converToONNX(deepModel, cfg):
    model = deepModel.eval().cuda()
    db = TUM_RGBD(cfg.INPUT.RESIZE, "data/tum2", r=2)

    trainloader = torch.utils.data.DataLoader(db, batch_size=1, shuffle=False, num_workers=8)
    for i, data in enumerate(trainloader, 0):
        if i> 0:
            break
        images_batch, poses_batch, gt_batch, filled_batch, pred_batch, intrinsics_batch, frame_id = data
        images = images_batch.permute(0, 1, 4, 2, 3)
        poses = poses_batch.cuda()
        images = images.float().cuda()
        intrinsics = intrinsics_batch.float().cuda()
        with torch.no_grad():
            print("hello")
            outputs = deepModel(poses, images, intrinsics)
            #print(outputs)
            torch.onnx.export(
                    deepModel,
                    (poses,
                    images, 
                    intrinsics),
                    "deep_onnx.onnx",
                    opset_version=12,
                    do_constant_folding=True,
                    input_names=["poses","images","intrinsics"],
                    output_names=["output"]
                    )
            #model_trt = torch2trt(model, [poses, images, intrinsics ])

if __name__ == '__main__':
    cfg = config.cfg_from_file("cfgs/tum_torch/tum_2_2_shufflev2_fast.yaml")
    deepModel = DepthModule(cfg)
    checkpoint = torch.load("pytorch_model/tum/shufflenetv2_fast/final.pth")
    deepModel.load_state_dict(checkpoint['net'])
    inference_test(deepModel, cfg)
    #converToTensorrt(deepModel,cfg)
    #converToONNX(deepModel,cfg)
# print(model)
# model.load_state_dict(torch.load('pytorch/tum/tmu_model/depth.pth'))


# for i in model.state_dict():
#     print(i)