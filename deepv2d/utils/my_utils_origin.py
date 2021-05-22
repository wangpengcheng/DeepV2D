
import torch
import numpy as np
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import random

from PIL import Image
import os

def add_depth_acc(gt, pr):
    """
    tensorborad 写入函数

    Args:
        gt ([type]): [description]
        pr ([type]): [description]
    """
    gt = torch.reshape(gt, [-1])
    pr = torch.reshape(pr, [-1])
    v = (gt > 0.1) & (gt < 10.0)
    gt = gt[v]
    pr = pr[v]
    # 筛选最大值
    thresh = torch.maximum(gt / pr, pr / gt)
    # 计算准确率平均值
    delta = torch.mean((thresh < 1.25).float())
    # 计算绝对值
    abs_rel = torch.mean(torch.abs(gt-pr) / gt)
    return delta, abs_rel


def set_random_seed(seed = 10, deterministic=False, benchmark=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
    if benchmark:
        torch.backends.cudnn.benchmark = True

def resize_crop(images, factor, inter):
    # 获取深度
    ht, wd = images.shape[-2:]
    # 计算新尺度
    ht1 = int(ht*factor)
    wd1 = int(wd*factor)
    # 计算差距值
    dx = (wd1 - wd) // 2 
    dy = (ht1 - ht) // 2
    # 图像缩放,裁剪
    images = torch.nn.functional.interpolate(images, scale_factor=factor, mode= inter)

    images = images[...,dy:dy+ht, dx:dx+wd]
    return images

def save_tensor(original_tensor, save_name):
    unloader = transforms.ToPILImage()
    image = original_tensor.cpu().clone()  # clone the tensor
    image = image.squeeze(0)  # remove the fake batch dimension
    
    image = unloader(image)
    image.save(save_name)



def scale(cfg, images, depth_gt, intrinsics, filled):
    """ Random scale augumentation """
    """
    根据缩放尺寸从中心进行裁剪
    Returns:
        [type]: [description]
    """
    if len(cfg.INPUT.SCALES) > 1:
        # 将scales转换为常量
        scales = cfg.INPUT.SCALES
        scale_ix = torch.Tensor(1).uniform_(0, len(cfg.INPUT.SCALES))
        scale_ix = scale_ix.int().to(intrinsics.device)
        scale_ix = int(scale_ix.int())
        # 索引筛选
        s = scales[scale_ix]
        # 计算宽度和高度
        ht = int(cfg.INPUT.HEIGHT*cfg.INPUT.RESIZE)
        wd = int(cfg.INPUT.WIDTH*cfg.INPUT.RESIZE)
        # 进行缩放
        ht1 = int(ht*s)
        # 进行缩放
        wd1 = int(wd*s)
        dx = (wd1 - wd) // 2 
        dy = (ht1 - ht) // 2
        # rgb图像缩放
        images = resize_crop(images, s, 'bilinear')
        # 深度图像缩放
        depth_gt = resize_crop(depth_gt, s, 'nearest')
        # 图像缩放
        if filled is not None:
            filled = resize_crop(filled, s, 'nearest')
        # 相机内参
        intrinsics = (intrinsics * s) - torch.Tensor([0, 0, dx, dy]).to(intrinsics.device)

    return images, depth_gt, intrinsics, filled


def augument1(images):
    # randomly shift gamma
    # 随机shift变换
    images = transforms.ToPILImage()(images)
    loader = transforms.Compose([transforms.ToTensor()])
    random_gamma = torch.Tensor(1).uniform_(0.9, 1.1)
    #images = Image.open("data/tum2/rgbd_dataset_freiburg3_cabinet/rgb/1341841278.906584.png")
    
    images = TF.adjust_gamma(images, random_gamma)
    #save_tensor(images, "gamma.png")
    images.save("gamma.png")
    # 亮度变换
    images = transforms.ColorJitter(brightness=0.2)(images)
    images.save("brightness.png")
    # 对比度
    images = transforms.ColorJitter(contrast=0.2)(images)
    images.save("contrast.png")
    # 饱和度
    images = transforms.ColorJitter(saturation=0.2)(images)
    images.save("saturation.png")
    # 色相变换/颜色变换
    images = transforms.ColorJitter(hue=0.2)(images)
    images.save("hue.png")
    # 转换为Tensor
    images = transforms.ToTensor()(images)
    return images

def augument2(images):
    # gamma变换
    random_gamma = torch.Tensor(1).uniform_(0.9, 1.1).to(images.device)
    images = 255.0*((images/255.0)**random_gamma)
    # 亮度变换
    random_brightness = torch.Tensor(1).uniform_(0.8, 1.2).to(images.device)
    images *= random_brightness
    # 颜色随机值
    random_colors = torch.Tensor(3).uniform_(0.8, 1.2).to(images.device).view(1, 3, 1, 1)
    images *= random_colors
    images = torch.clamp(images, 0.0, 255.0)
    return images


def augument1(images):
    # randomly shift gamma
    # 随机shift变换
    images = transforms.ToPILImage()(images)
    #images.save("origin.png")
    random_gamma = torch.Tensor(1).uniform_(0.8, 1.2)
    #images = Image.open("data/tum2/rgbd_dataset_freiburg3_cabinet/rgb/1341841278.906584.png")
    images = TF.adjust_gamma(images, random_gamma)
    #save_tensor(images, "gamma.png")
    #images.save("gamma.png")
    # 亮度随机值
    ra  = np.random.uniform(0.8, 1.2, 3)
    r_he = np.random.uniform(0.3, 0.5)
    # 亮度变换
    images = transforms.ColorJitter(brightness=ra[0],
                                    contrast=ra[1],
                                    saturation=ra[2], 
                                    hue=r_he
                                    )(images)
    # #images.save("brightness.png")
    # # 对比度
    # images = transforms.ColorJitter(contrast=0.3)(images)
    # #images.save("contrast.png")
    # # 饱和度
    # images = transforms.ColorJitter(saturation=0.3)(images)
    # #images.save("saturation.png")
    # # 色相变换/颜色变换
    # images = transforms.ColorJitter(hue=0.3)(images)
    #images.save("res.png")
    # 转换为Tensor
    images = transforms.ToTensor()(images)*255.0
    return images



def prepare_inputs(cfg, images, depth, intrinsics, filled=None):
    images = torch.from_numpy(images)
    depth = torch.from_numpy(depth)
    #print(depth.shape)
    intrinsics = torch.from_numpy(intrinsics)
    images = images.permute(0, 3, 1, 2)
    n, c, w, h = images.shape[:]
    #images = images.view(n, c, w, h)
    #print(images.shape)
    images = augument2(images)
    depth = depth.view(1, 1, w, h)
    images, depth, intrinsics, filled = scale(cfg, images, depth, intrinsics, filled)
    # print(images.shape)
    # print(depth.shape)
    #images = images.view(b, n, c, w, h)
    depth = depth.view(w, h)
    return images, depth,  intrinsics ,filled