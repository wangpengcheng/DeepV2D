import PIL.Image as Image
import torchvision.transforms as transforms
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
 
#读入图片
image=Image.open("/home/node/workspace/DeepV2D/depth.png")
print(image.size, image.format, image.mode)
w,h = image.size[0:2]


w=int(w/2)
h=int(h/2)
print(w,h)


#生成一个CenterCrop类的对象,用来将图片从中心裁剪成224*224
transform1 = transforms.Compose([
        #transforms.CenterCrop(image.size),
        Resize((480,640))
])

image = transform1(image)  
print(image.size, image.format, image.mode)

#将裁剪之后的图片保存下来
image.save("test.png", format='PNG')
