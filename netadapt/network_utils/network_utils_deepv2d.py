from typing import Tuple
from .network_utils_abstract import NetworkUtilsAbstract
from collections import OrderedDict

import os
import sys
sys.path.append("/home/node/workspace/DeepV2D/deepv2d")
sys.path.append("/home/node/workspace/DeepV2D/netadapt")
import copy
import time
import torch
import pickle
import warnings
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data.sampler as sampler

print(sys.path)
from deepv2d.data_stream.tum import TUM_RGBD
from deepv2d.modules.my_loss import LightLoss
from deepv2d.utils.my_utils import *
from constants import *
import functions as fns

'''
    This is an example of NetAdapt applied to AlexNet.
    We measure the latency on GPU.
'''

'''
    The size of feature maps of simplified layers along channel dimmension 
    are multiples of '_MIN_FEATURE_SIZE'.
    The reason is that on mobile devices, the computation of (B, 7, H, W) tensors 
    would take longer time than that of (B, 8, H, W) tensors.
'''
_MIN_CONV_FEATURE_SIZE = 16
_MIN_FC_FEATURE_SIZE   = 64

'''
    How many times to run the forward function of a layer in order to get its latency.
'''
_MEASURE_LATENCY_SAMPLE_TIMES = 500

'''
    The batch size of input data when running forward functions to measure latency.
'''
_MEASURE_LATENCY_BATCH_SIZE = 128

class networkUtils_deepv2d(NetworkUtilsAbstract):
    num_simplifiable_blocks = None
    input_data_shape = None
    train_loader = None
    holdout_loader = None
    val_loader = None
    optimizer = None

    def __init__(self, model, input_data_shape, dataset_path, finetune_lr=1e-3):
        '''
            Initialize:
                (1) network definition 'network_def'
                (2) num of simplifiable blocks 'num_simplifiable_blocks'. 
                (3) loss function 'criterion'
                (4) data loader for training/validation set 'train_loader' and 'holdout_loader',
                
            Need to be implemented:
                (1) finetune/evaluation data loader
                (2) loss function
                (3) optimizer
                
            Input: 
                `model`: model from which we will get network_def.
                `input_data_shape`: (list) [C, H, W].
                `dataset_path`: (string) path to dataset.
                `finetune_lr`: (float) short-term fine-tune learning rate.
        '''
        
        super().__init__()
        self.model = model
        # Set the shape of the input data.
        # 设置输入数据
        self.input_data_shape = input_data_shape
        # Set network definition (conv & fc)
        # 获取网络定义模型
        network_def = self.get_network_def_from_model(model)        
        # Set num_simplifiable_blocks.
        self.num_simplifiable_blocks = 0
        # 统计所有非深度可分离卷积
        for layer_name, layer_properties in network_def.items():
            if not layer_properties[KEY_IS_DEPTHWISE]:
                self.num_simplifiable_blocks += 1                
        # We cannot reduce the number of filters in the output layer (1).
        # also not consider simplifying the last two FC layer
        self.num_simplifiable_blocks -= 1  

        '''
            The following variables need to be defined depending on tasks:
                (1) finetune/evaluation data loader
                (2) loss function
                (3) optimizer
        '''
        # Data loaders for fine tuning and evaluation.
        self.batch_size = 4
        self.num_workers = 6
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.finetune_lr = finetune_lr
        
        # 构建数据集
        train_dataset = TUM_RGBD(0.5, dataset_path)
       
        # 创建dataloader
        train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=self.batch_size, 
        num_workers=self.num_workers, 
        pin_memory=True, shuffle=True)#, sampler=train_sampler)
        self.train_loader = train_loader
        
        val_dataset =  TUM_RGBD(0.5, dataset_path, test=True)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers, 
            pin_memory=True
        ) #, sampler=valid_sampler)
        self.val_loader = val_loader   
        
        self.criterion = LightLoss()
        
        
    def _get_layer_by_param_name(self, model, param_name):
        '''
            please refer to def get_layer_by_param_name(...) in functions.py
        '''

        return fns.get_layer_by_param_name(model, param_name)

    def _get_keys_from_ordered_dict(self, ordered_dict):
        '''
            please refer to def get_keys_from_ordered_dict(...) in functions.py
        '''
        return fns.get_keys_from_ordered_dict(ordered_dict)
    

    def get_network_def_from_model(self, model):
        '''
            please refer to get_network_def_from_model(...) in functions.py
        '''
        
        return fns.get_network_def_from_model(model, self.input_data_shape)
    
    
    def simplify_network_def_based_on_constraint(self, 
                                                network_def, 
                                                block, 
                                                constraint, 
                                                resource_type,
                                                lookup_table_path=None
                                                ):
        '''
            Derive how much a certain block of layers ('block') should be simplified 
            based on resource constraints.
            
            Here we treat one block as one layer although a block can contain several layers.
            
            Input:
                `network_def`: simplifiable network definition (conv & fc). Get network def from self.get_network_def_from_model(...)
                `block`: (int) index of block to simplify
                `constraint`: (float) representing the FLOPs/weights/latency constraint the simplied model should satisfy
                `resource_type`: `FLOPs`, `WEIGHTS`, or `LATENCY`
                `lookup_table_path`: (string) path to latency lookup table. Needed only when resource_type == 'LATENCY'
        
            Output:
                `simplified_network_def`: simplified network definition. Indicates how much the network should
                be simplified/pruned.
                `simplified_resource`: (float) the estimated resource consumption of simplified models.
        '''
        
        return fns.simplify_network_def_based_on_constraint(network_def, 
                                                            block, 
                                                            constraint, 
                                                            resource_type, 
                                                            lookup_table_path
                                                            )
        

    def simplify_model_based_on_network_def(self, simplified_network_def, model):
        '''
            Choose which filters to perserve
            
            Here filters with largest L2 magnitude will be kept
            
            please refer to def simplify_model_based_on_network_def(...) in functions.py
        '''
        
        return fns.simplify_model_based_on_network_def(simplified_network_def, model)
    
    
    def extra_history_info(self, network_def):
        '''
            return # of output channels per layer
            
            Input: 
                `network_def`: (dict)
            
            Output:
                `num_filters_str`: (string) show the num of output channels for each layer
        '''
        num_filters_str = [str(layer_properties[KEY_NUM_OUT_CHANNELS]) for _, layer_properties in
                               network_def.items()]
        num_filters_str = ' '.join(num_filters_str)
        return num_filters_str
    

    def _compute_weights_and_flops(self, network_def):
        '''
            please refer to def compute_weights_and_macs(...) in functions.py
        '''
        return fns.compute_weights_and_macs(network_def)
    
    
    def _compute_latency_from_lookup_table(self, network_def, lookup_table_path):
        '''
            please refer to def compute_latency_from_lookup_table(...) in functions.py
        '''
        return fns.compute_latency_from_lookup_table(network_def, lookup_table_path)

    
    def build_lookup_table(self, network_def_full, resource_type, lookup_table_path, 
                           min_conv_feature_size=_MIN_CONV_FEATURE_SIZE, 
                           min_fc_feature_size=_MIN_FC_FEATURE_SIZE, 
                           measure_latency_batch_size=_MEASURE_LATENCY_BATCH_SIZE, 
                           measure_latency_sample_times=_MEASURE_LATENCY_SAMPLE_TIMES, 
                           verbose=True):
        # Build lookup table for latency
        '''
            please refer to def build_latency_lookup_table(...) in functions.py
        '''
        return fns.build_latency_lookup_table(network_def_full, lookup_table_path,
                                      min_conv_feature_size=min_conv_feature_size, 
                                      min_fc_feature_size=min_fc_feature_size,
                                      measure_latency_batch_size=measure_latency_batch_size,
                                      measure_latency_sample_times=measure_latency_sample_times,
                                      verbose=verbose)
        
        
    def compute_resource(self, network_def, resource_type, lookup_table_path=None):
        '''
            please refer to def compute_resource(...) in functions.py
        '''
        return fns.compute_resource(network_def, resource_type, lookup_table_path)
    
    
    def get_num_simplifiable_blocks(self):
        return self.num_simplifiable_blocks
    

    def fine_tune(self, model, iterations, print_frequency=100):
        '''
            short-term fine-tune a simplified model
            
            Input:
                `model`: model to be fine-tuned.
                `iterations`: (int) num of short-term fine-tune iterations.
                `print_frequency`: (int) how often to print fine-tune info.
            
            Output:
                `model`: fine-tuned model.
        '''
        
        _NUM_CLASSES = 10
        optimizer = torch.optim.SGD(model.parameters(), self.finetune_lr, 
                                         momentum=self.momentum, weight_decay=self.weight_decay)
        model = model.cuda()
        model.train()
        dataloader_iter = iter(self.train_loader)

        for i in range(iterations):
            try:
                (images_batch, poses_batch, gt_batch, myfilled, myfilled, intrinsics_batch, frameid) = next(dataloader_iter)
            except:
                dataloader_iter = iter(self.train_loader)
                (images_batch, poses_batch, gt_batch, myfilled, myfilled, intrinsics_batch, frameid) = next(dataloader_iter)
            
            images_batch = images_batch.permute(0, 1, 4, 2, 3)
            Ts = poses_batch.cuda()
            images = images_batch.cuda()
            intrinsics = intrinsics_batch.cuda().float()
            gt = gt_batch.cuda()

            if i % print_frequency == 0:
                print('Fine-tuning iteration {}'.format(i))
                sys.stdout.flush()
            
            

            pred = model(Ts, 
                    images, 
                    intrinsics)
            loss = self.criterion(gt, pred)
            optimizer.zero_grad()
            loss.backward()  # compute gradient and do SGD step
            optimizer.step()
        return model
    

    def evaluate(self, model, print_frequency=10):
        '''
            Evaluate the accuracy of the model
            
            Input:
                `model`: model to be evaluated.
                `print_frequency`: how often to print evaluation info.
                
            Output:
                accuracy: (float) (0~100)
        '''
        
        model = model.cuda()
        model.eval()
        acc = .0
        num_samples = .0
        with torch.no_grad():
            for i, data in enumerate(self.val_loader, 0):
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
                temp_delta, temp_abs_rel =  add_depth_acc(gt.detach(), pred.detach())
                acc += float(temp_delta.item())
                num_samples += 1
                
                if i % print_frequency == 0:
                    fns.update_progress(i, len(self.val_loader))
                    print(' ')
        print(' ')
        print('Test accuracy: {:4.2f}% '.format(float(acc/num_samples*100)))
        print('===================================================================')
        return acc/num_samples*100
    

def deepv2d(model, input_data_shape, dataset_path, finetune_lr=1e-3):
    return networkUtils_deepv2d(model, input_data_shape, dataset_path, finetune_lr)