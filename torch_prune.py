import sys, os

sys.path.append('deepv2d')
sys.path.append('evaluation')

from shutil import copyfile
import torch
import time
import cv2
import matplotlib.pyplot as plt
from deepv2d.modules.depth_module import DepthModule
from deepv2d.modules.networks.coders import *
from deepv2d.modules.networks.layer_ops import *
from deepv2d.data_stream.tum import TUM_RGBD
from deepv2d.modules.my_loss import LightLoss
from deepv2d.utils.my_utils import *

import torch_pruning as tp
import argparse
import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn 
import numpy as np 
from thop import profile
from thop import clever_format
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, required=True, choices=['train', 'prune', 'test'])
parser.add_argument('--lr', type=float, default=0.00001)
parser.add_argument('--dataset_path', type=str, default="data/tum3")
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--verbose', action='store_true', default=True)
parser.add_argument('--total_epochs', type=int, default=100)
parser.add_argument('--step_size', type=int, default=500)
parser.add_argument('--round', type=int, default=2)

args = parser.parse_args()



def evaluate(model, val_loader, print_frequency=10):
        '''
            Evaluate the accuracy of the model
            
            Input:
                `model`: model to be evaluated.
                `print_frequency`: how often to print evaluation info.
                
            Output:
                accuracy: (float) (0~100)
        '''
        
        #model = model.cuda()
        model.eval()
        acc = .0
        num_samples = .0
        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                images_batch, poses_batch, gt_batch, myfilled, myfilled, intrinsics_batch, frameid = data
                images_batch = images_batch.permute(0, 1, 4, 2, 3)
                Ts = poses_batch.cuda()
                images = images_batch.cuda()
                intrinsics = intrinsics_batch.cuda().float()
                gt = gt_batch.cuda()
                pred = model(
                    Ts,
                    images,
                    intrinsics
                    )
                temp_delta, temp_abs_rel =  add_depth_acc(
                                                gt.detach(), 
                                                pred.detach()
                                            )
                acc += float(temp_delta.item())
                num_samples += 1
               
        print(' ')
        print('Test accuracy: {:4.2f}% '.format(float(acc/num_samples*100)))
        print('===================================================================')
        return acc/num_samples*100



def train_model(model, train_loader, test_loader):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, 0.1)
    loss_function = LightLoss()
    model.to(device)
    log_fps = 10
    best_acc = -1
    
    for epoch in range(args.total_epochs):
        model.train()
        for i, data in enumerate(train_loader):
            (images_batch, poses_batch, gt_batch, myfilled, myfilled, intrinsics_batch, frameid) = data
            images_batch = images_batch.permute(0, 1, 4, 2, 3)
            Ts = poses_batch.to(device)
            images = images_batch.to(device)
            intrinsics = intrinsics_batch.float().to(device)
            gt = gt_batch.to(device)

            pred = model(
                    Ts, 
                    images, 
                    intrinsics
                    )
            
            loss = loss_function(gt, pred)
            optimizer.zero_grad()
            loss.backward()  # compute gradient and do SGD step
            optimizer.step()
            # 输出日志
            if epoch  % log_fps == 0 and args.verbose:
                print("Epoch %d/%d, iter %d/%d, loss=%.4f" % (epoch, args.total_epochs, i, len(train_loader), loss.item()))
                # 进行测试
                acc = evaluate(model, test_loader)
                print("Epoch %d/%d, Acc=%.4f" % (epoch, args.total_epochs, acc))
                if best_acc < acc:
                    torch.save( model, 'deepv2d-round%d.pth.tar'%(args.round) )
                    best_acc=acc
        scheduler.step()
    print("Best Acc=%.4f"%(best_acc))
    return best_acc

def prune_model(model, input, pre = 0.4):
    model.cpu()
    DG = tp.DependencyGraph().build_dependency( model, input)
    def prune_conv(conv, amount=0.2):
        #weight = conv.weight.detach().cpu().numpy()
        #out_channels = weight.shape[0]
        #L1_norm = np.sum( np.abs(weight), axis=(1,2,3))
        #num_pruned = int(out_channels * pruned_prob)
        #pruning_index = np.argsort(L1_norm)[:num_pruned].tolist() # remove filters with small L1-Norm
        strategy = tp.strategy.L1Strategy()
        pruning_index = strategy(conv.weight, amount=amount)
        plan = DG.get_pruning_plan(conv, tp.prune_conv, pruning_index)
        plan.exec()
        
    for m in model.modules():
        if isinstance( m, Shufflenetv2Encoder):
            #prune_conv( m.conv1.conv, pre )
            for m1 in m.modules():
                if isinstance( m1, ShuffleNetUnitV2A ):
                    prune_conv(m1.conv1.conv, pre)
                    prune_conv(m1.depthwise_conv_bn.point_wise, pre)
                elif isinstance( m1, ShuffleNetUnitV2B ):
                    prune_conv(m1.conv1.conv, pre)
                    prune_conv(m1.depthwise_conv_bn.point_wise, pre)
                    prune_conv(m1.depthwise_conv_bn_01.point_wise, pre)
                elif isinstance( m1, Aspp2d ):
                    prune_conv( m1.conv1.conv, pre)
                    prune_conv( m1.dilate_conv1.conv, pre)
                    prune_conv( m1.dilate_conv2.conv, pre)
                    prune_conv( m1.dilate_conv3.conv, pre)
                elif isinstance( m1, Conv2dBnRel ):
                     #prune_conv( m1.conv, pre)
                     print("no thing")
                else:
                    print(type(m1))
        elif isinstance( m, FastResnetDecoderAvg):
            print("no thing to do !")

    return model

def test_model(model, input):
    macs, params = profile(model, inputs=tuple(input))
    macs, params = clever_format([macs, params], "%.4f")
    print(macs)
    print(params)

def main():
    #train_loader, test_loader = get_dataloader()
    if args.mode=='train':
        args.round=0
        model = ResNet18(num_classes=10)
        train_model(model, train_loader, test_loader)
    elif args.mode=='prune':
        previous_ckpt = "netadapt/models/deepv2d/model.pth.tar"
        print("Pruning round %d, load model from %s"%( args.round, previous_ckpt ))
        model = torch.load( previous_ckpt )
        
         # 构建数据集
        train_dataset = TUM_RGBD(0.5,  args.dataset_path)
        num_workers = 12 

        # 创建dataloader
        train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size= args.batch_size, 
        num_workers=num_workers, 
        pin_memory=True, shuffle=True)#, sampler=train_sampler)
        
        val_dataset =  TUM_RGBD(0.5, args.dataset_path, test=True)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size= args.batch_size, 
            shuffle=False,
            num_workers=num_workers, 
            pin_memory=True
        ) #, sampler=valid_sampler)
        # 
        test_loader = val_loader
        # 进行剪枝
        inputs = []
        inputs.append(torch.randn(1, 5, 4, 4))
        inputs.append(torch.randn(1, 5, 3, 240, 320))
        inputs.append(torch.randn(1, 4))
        params1 = sum([np.prod(p.size()) for p in model.parameters()])
        # 设置不同裁剪率 返回最优模型
        max_len = 10
        pres = [0.2, 0.3, 0.4, 0.5, 0.7, 0.8]
        #pres = [0.8, 0.7, 0.5, 0.4, 0.3, 0.2]
        for temp_pre in pres:
            
            model  = prune_model(model, inputs, pre = temp_pre)
            #test_model(model, inputs)
            #print(model)
            params = sum([np.prod(p.size()) for p in model.parameters()])
            
            print("Number of origin Parameters: %.5fM"%(params1/1e6))
            print("Number of Parameters: %.5fM"%(params/1e6))

            best_acc = train_model(model, train_loader, test_loader)
            # 进行模型保存
            if best_acc > 80.0:
                print("pre: {} best_acc {}% ".format(temp_pre, best_acc))
                origin_name = "deepv2d-round{}.pth.tar".format(args.round)
                model_name = "pre_{}_acc_{}.pth.tar".format(temp_pre, best_acc)
                os.popen("cp {} {}".format(origin_name, model_name))
            else:
                print("pre: {} best_acc {}% too low: 80%".format(temp_pre, best_acc))
    elif args.mode=='test':
        ckpt = 'resnet18-round%d.pth'%(args.round)
        print("Load model from %s"%( ckpt ))
        model = torch.load( ckpt )
        params = sum([np.prod(p.size()) for p in model.parameters()])
        print("Number of Parameters: %.1fM"%(params/1e6))
        acc = eval(model, test_loader)
        print("Acc=%.4f\n"%(acc))

if __name__=='__main__':
    main()