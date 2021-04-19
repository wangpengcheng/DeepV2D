
import torch
import numpy as np
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import random

from PIL import Image
import os

def set_random_seed(seed = 10, deterministic=False, benchmark=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
    if benchmark:
        torch.backends.cudnn.benchmark = True

def resize_crop(images, scale_factor, inter):
    # 获取深度
    ht, wd = images.shape[-2:]
    # 计算新尺度
    ht1 = int(ht*scale_factor)
    wd1 = int(wd*scale_factor)
    # 计算差距值
    dx = (wd1 - wd) // 2 
    dy = (ht1 - ht) // 2
    # 图像缩放
    images = transforms.functional.resize(images, (ht1, wd1), interpolation=inter)
    # 对图像进行裁剪
    images = transforms.functional.crop(images, dy, dx, ht, wd)
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
        images = resize_crop(images, s, Image.BILINEAR)
        # 深度图像缩放
        depth_gt = resize_crop(depth_gt, s, Image.NEAREST)
        # 图像缩放
        if filled is not None:
            filled = resize_crop(filled, s, Image.NEAREST)
        # 相机内参
        intrinsics = (intrinsics * s) - torch.Tensor([0, 0, dx, dy])

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

def augument(images):
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
    images = transforms.ToTensor()(images)*255
    return images



def prepare_inputs(cfg, images, depth, intrinsics, filled=None):
    
    b,n,c,w,h = images.shape[:]
    images = images.view(b*n, c, w, h)
    #save_tensor(images[0], "data_origin.png")
    # for i in range(len(images)):
    #     temp = augument(images[i])
    #     images[i] = temp
    images, depth, intrinsics, filled = scale(cfg, images, depth, intrinsics, filled = None)
    images = images.view(b, n, c, w, h)
    return images, depth,  intrinsics, filled