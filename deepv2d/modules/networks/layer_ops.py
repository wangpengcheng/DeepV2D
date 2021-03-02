import torch
import torch.nn as nn



class Conv2d(nn.Module):
    '''
    This class is for a convolutional layer.
    C
    '''
    def __init__(self, nIn, nOut, kSize, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        '''
        super().__init__()
        # 自适应边长填充
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=True)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output

class MaxPool2D(nn.Module):

    def __init__(self, nIn, nOut, kSize, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optinal stide for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1)/2)
        self.pool = nn.MaxPool2d(kSize, kSize, padding=padding)
    
    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.pool(input)
        return output

class MaxPool3D(nn.Module):

    def __init__(self, nIn, nOut, kSize, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optinal stide for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1)/2)
        self.pool = nn.MaxPool3d(kSize, padding=padding)
    
    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.pool(input)
        return output

class Conv2dBN(nn.Module):
    '''
       This class groups the convolution and batch normalization
       C+B
    '''
    def __init__(self, nIn, nOut, kSize, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optinal stide for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=True)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-05)

    def forward(self, input):
        '''

        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        output = self.bn(output)
        return output

class Conv2dBnRel(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    c+BN+RL
    '''
    def __init__(self, nIn, nOut, kSize, stride=1):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1)/2)
        #self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=True)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=True)
        #self.conv1 = nn.Conv2d(nOut, nOut, (1, kSize), stride=1, padding=(0, padding), bias=True)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-05)
        self.act = nn.ReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        #output = self.conv1(output)
        output = self.bn(output)
        output = self.act(output)
        return output



class BR2d(nn.Module):
    '''
        This class groups the batch normalization and PReLU activation
        BN+LU
    '''
    def __init__(self, nOut):
        '''
        :param nOut: output feature maps
        '''
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut, eps=1e-05)
        self.act = nn.ReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: normalized and thresholded feature map
        '''
        output = self.bn(input)
        output = self.act(output)
        return output


class Conv3d(nn.Module):
    '''
    This class is for a convolutional layer.
    3d卷积
    '''
    def __init__(self, nIn, nOut, kSize, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        pytorch 
        in N, Ci, D, H, W
        out N, Co, D, H, W
        tensorflow
        [batch, in_depth, in_height, in_width, in_channels] N,D,H,W,C
        '''
        super().__init__()
        # 自适应边长填充
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv3d(nIn, nOut, (kSize, kSize, kSize), stride=stride, padding=(padding, padding, padding), bias=True)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output

class Conv3dBN(nn.Module):
    '''
       This class groups the convolution and batch normalization
       C+B
    '''
    def __init__(self, nIn, nOut, kSize, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optinal stide for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv3d(nIn, nOut, (kSize, kSize, kSize), stride=stride, padding=(padding, padding, padding), bias=True)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-05)

    def forward(self, input):
        '''

        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        output = self.bn(output)
        return output

class Conv3dBnRel(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    c+BN+RL
    '''
    def __init__(self, nIn, nOut, kSize, stride=1):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv3d(nIn, nOut, (kSize, kSize, kSize), stride=stride, padding=(padding, padding, padding), bias=True)
        self.bn = nn.BatchNorm3d(nOut, eps=1e-05)
        self.act = nn.ReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        #output = self.conv1(output)
        output = self.bn(output)
        output = self.act(output)
        return output

class BR3d(nn.Module):
    '''
        This class groups the batch normalization and PReLU activation
        BN+LU
    '''
    def __init__(self, nOut):
        '''
        :param nOut: output feature maps
        '''
        super().__init__()
        self.bn = nn.BatchNorm3d(nOut, eps=1e-05)
        self.act = nn.ReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: normalized and thresholded feature map
        '''
        output = self.bn(input)
        output = self.act(output)
        return output


class CDilated(nn.Module):
    '''
    This class defines the dilated convolution.
    空洞卷积
    '''
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        padding = int((kSize - 1)/2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=True, dilation=d)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output

class ResConv2d(nn.Module):
    def __init__(self, nIn, nOut, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        # 进行初始化
        super().__init__()
        if stride==1:
            self.conv1 = Conv2dBnRel(nIn, nOut, 3, 1)
            self.conv2 = Conv2dBnRel(nOut, nOut, 3, 1)
            self.conv3 = None
        else:
            self.conv1 = Conv2dBnRel(nIn, nOut, 3, 1)
            self.conv2 = Conv2dBnRel(nOut, nOut, 3, 2)
            self.conv3 = Conv2dBnRel(nIn, nOut, 1, 2)
    
            
    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        out = self.conv1(input)
        out = self.conv2(out)
        if self.conv3 is not None:
            residual = self.conv3(input)
            out += residual
        else:
            out += input
        return out


class ResConv3d(nn.Module):
    def __init__(self, nIn, nOut, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        # 进行初始化
        super().__init__()
        if stride==1:
            self.conv1 = Conv3dBnRel(nIn, nOut, 3, 1)
            self.conv2 = Conv3dBnRel(nOut, nOut, 3, 1)
            self.conv3 = None
        else:
            self.conv1 = Conv3dBnRel(nIn, nOut, 3, 1)
            self.conv2 = Conv3dBnRel(nOut, nOut, 3, 2)
            self.conv3 = Conv3dBnRel(nIn, nOut, 1, 2)
    
            
    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        out = self.conv1(input)
        out = self.conv2(out)
        if self.conv3 is not None:
            residual = self.conv3(input)
            out += residual
        else:
            out += input
        return out
        
class FastResConv2d(nn.Module):
    def __init__(self, nIn, nOut, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        # 进行初始化
        super().__init__()
        if stride==1:
            self.conv1 = Conv2dBnRel(nIn, nOut, 1, 1)
            self.conv2 = Conv2dBnRel(nOut, nOut, 3, 1)
            self.conv3 = Conv2dBnRel(nOut, nOut, 1, 1)
            self.conv4 = None
        else:
            self.conv1 = Conv2dBnRel(nIn, nOut, 1, 1)
            self.conv2 = Conv2dBnRel(nOut, nOut, 3, 2)
            self.conv3 = Conv2dBnRel(nOut, nOut, 1, 1)
            self.conv4 = Conv2dBnRel(nIn, nOut, 1, 2)
    
            
    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.conv4  is not None:
            residual = self.conv4(input)
            out += residual
        else:
            out += input
        return out


class FastResConv3d(nn.Module):
    def __init__(self, nIn, nOut, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        # 进行初始化
        super().__init__()
        if stride ==1:
            self.conv1 = Conv3dBnRel(nIn, nOut, 1, 1)
            self.conv2 = Conv3dBnRel(nOut, nOut, 3, 1)
            self.conv3 = Conv3dBnRel(nOut, nOut, 1, 1)
            self.conv4 = None
        else:
            self.conv1 = Conv3dBnRel(nIn, nOut, 1, 1)
            self.conv2 = Conv3dBnRel(nOut, nOut, 3, 2)
            self.conv3 = Conv3dBnRel(nOut, nOut, 1, 1)
            self.conv4 = Conv3dBnRel(nIn, nOut, 1, 2)
    
            
    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.conv4  is not None:
            residual = self.conv4(input)
            out += residual
        else:
            out += input
        return out

