
from geometry.intrinsics import *
from PIL import Image
import matplotlib.pyplot as plt
from geometry.projective_ops import *
from geometry.se3 import *

import torch 

def test_rescale():
    image=Image.open("/home/node/workspace/DeepV2D/depth.png").convert('RGB')
    print(image.size, image.format, image.mode)
    tensorSrc = transforms.ToTensor()(image)
    tensorSrc = torch.stack((tensorSrc,tensorSrc),dim=0)
    b = torch.randn(3,3)
    
    image,ins = rescale_depths_and_intrinsics(tensorSrc,b)
    
    image  = transforms.ToPILImage()(image[0])
    print(image.size, image.format, image.mode)
    #将裁剪之后的图片保存下来
    image.save("test.png", format='PNG')

def test_projective():
    # image=Image.open("/home/node/workspace/DeepV2D/depth.png").convert('RGB')
    # print(image.size, image.format, image.mode)
    # tensorSrc = transforms.ToTensor()(image)
    #tensorSrc = torch.stack((tensorSrc,tensorSrc),dim=0)
    a = torch.randn([100,640,480,3])
    b = torch.randn(100,4,4)
    #coords =  project(a,b)
    coo1 = backproject(a,b)

def test_se3():
    # image=Image.open("/home/node/workspace/DeepV2D/depth.png").convert('RGB')
    # print(image.size, image.format, image.mode)
    # tensorSrc = transforms.ToTensor()(image)
    #tensorSrc = torch.stack((tensorSrc,tensorSrc),dim=0)
    a = torch.randn(3,3)
    b = torch.randn(1,3)
    #coords =  project(a,b)
    coo1 = matdotv(a,b)
    print(coo1)
if __name__ == '__main__':
    test_se3()