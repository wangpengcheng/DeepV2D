# -*- coding: utf-8 -*-
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
import datetime
#from torch2trt import torch2trt
import torch_pruning as tp
from thop import profile

def inference_test(deepModel, cfg):
   
    #deepModel = deepModel.load_state_dict(torch.load('pytorch/tum/tmu_model/depth.pth'))
    
    db = TUM_RGBD(cfg.INPUT.RESIZE, "data/tum-01",test=True, r=2)

    trainloader = torch.utils.data.DataLoader(db, batch_size=1, shuffle=False, num_workers=1)
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
            print(images.shape)
            print(Ts.shape)
            print(intrinsics_batch.shape)
            # 计算时间

            time_start = datetime.datetime.now()
            outputs = deepModel(Ts, images, intrinsics_batch)
            time_end = datetime.datetime.now()
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
            result_out_dir = "{}/{}".format("data/tum3/rgbd_dataset_freiburg3_cabinet", "inference_result_2")
            # 检测路径文件夹
            if not os.path.exists(result_out_dir):
                os.makedirs(result_out_dir)
            cv2.imwrite("{}/{}.png".format(result_out_dir, i), image_depth)
            print("wirte image:{}/{}.png".format(result_out_dir,i))
            if i != 0:
                time_sum = time_sum + (time_end-time_start).total_seconds()
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
    db = TUM_RGBD(cfg.INPUT.RESIZE, "data/tum-02", r=2)

    trainloader = torch.utils.data.DataLoader(db, batch_size=1, shuffle=False, num_workers=1)
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

def get_all_macs(model, inputs):
    macs = 0,
    params = 0,
    macs, params = profile(model, inputs=inputs)
    for mm in model.modules():
        temp_macs, temp_params = get_all_macs(model, inputs=inputs)
        macs += temp_macs
        params += temp_params
    return macs, params

if __name__ == '__main__':
    cfg = config.cfg_from_file("cfgs/tum_torch/tum_2_2_shufflev2_fast.yaml")
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.TRAIN.USE_GPU
    #deepModel = DepthModule(cfg)
    # 先把模型改过来
    #deep_model_dict = deepModel.state_dict()
    # print(deepModel)
    #checkpoint = torch.load("pytorch_model/tum/shufflenetv2_fast/step_600.pth")
    #deepModel.load_state_dict(checkpoint['net'])
    #torch.save(deepModel, "netadapt/models/deepv2d/model.pth.tar")
    # 加载深度模型
    model = torch.load("netadapt/models/deepv2d/model.pth.tar")
   
    # 加载共同词典
    # pretrained_dict =  { k: v for k, v in model.state_dict().items() if k in deep_model_dict}
    # weight_set = {
    #     'encoder.res_conv1.conv1.conv.weight',
    #     'encoder.res_conv2.conv1.conv.weight',
    #     'encoder.res_conv4.conv1.conv.weight',
    #     'encoder.res_conv5.conv1.conv.weight',
    #     'encoder.res_conv7.conv1.conv.weight',
    #     'encoder.res_conv8.conv1.conv.weight',
    # }
    # # 遍历所有层并进行修改
    # for weight_layer in weight_set:
    #     pre_tensor = pretrained_dict[weight_layer]
    #     origin_tensor = deep_model_dict[weight_layer]
    #     n, c, w, h = list(origin_tensor.shape)
    #     pretrained_dict[weight_layer] = pre_tensor[0:n,0:c,0:w,0:h]

    #deep_model_dict.update(pretrained_dict)
    #deepModel.load_state_dict(deep_model_dict)
    #torch.save(deepModel, "netadapt/models/deepv2d/model.pth.tar")
    strategy = tp.strategy.L1Strategy() 
    DG = tp.DependencyGraph()
    inputs = []
    inputs.append(torch.randn(1, 5, 4, 4))
    inputs.append(torch.randn(1, 5, 3, 240, 320))
    inputs.append(torch.randn(1, 4))
   
    print(model)
    #macs, params = get_all_macs(model, inputs=inputs)

    DG.build_dependency(model, example_inputs = inputs)
    pruning_idxs = strategy(model.encoder.conv2.conv.weight, amount=0.5) # or manually selected pruning_idxs=[2, 6, 9]
    pruning_plan = DG.get_pruning_plan( model.encoder.conv2.conv, tp.prune_conv, idxs=pruning_idxs )
    print(pruning_plan)

    # 4. execute this plan (prune the model)
    pruning_plan.exec()

    print(model)
    
    #print(model_dict.cfg)
    #inference_test(deepModel, cfg)
    #converToTensorrt(deepModel,cfg)
    #converToONNX(deepModel,cfg)
# print(model)
# model.load_state_dict(torch.load('pytorch/tum/tmu_model/depth.pth'))


# for i in model.state_dict():
#     print(i)