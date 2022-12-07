## ------------------------------------------------------------------------------------------
## Confusing image quality assessment: Towards better augmented reality experience
## Huiyu Duan, Xiongkuo Min, Yucheng Zhu, Guangtao Zhai, Xiaokang Yang, and Patrick Le Callet
## IEEE Transactions on Image Processing (TIP)
## ------------------------------------------------------------------------------------------

from collections import namedtuple
import torch
from torchvision import models as tv


class squeezenet(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(squeezenet, self).__init__()
        pretrained_features = tv.squeezenet1_1(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.slice6 = torch.nn.Sequential()
        self.slice7 = torch.nn.Sequential()
        self.N_slices = 7
        for x in range(2):
            self.slice1.add_module(str(x), pretrained_features[x])
        for x in range(2,5):
            self.slice2.add_module(str(x), pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), pretrained_features[x])
        for x in range(10, 11):
            self.slice5.add_module(str(x), pretrained_features[x])
        for x in range(11, 12):
            self.slice6.add_module(str(x), pretrained_features[x])
        for x in range(12, 13):
            self.slice7.add_module(str(x), pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        h = self.slice6(h)
        h_relu6 = h
        h = self.slice7(h)
        h_relu7 = h
        vgg_outputs = namedtuple("SqueezeOutputs", ['relu1','relu2','relu3','relu4','relu5','relu6','relu7'])
        out = vgg_outputs(h_relu1,h_relu2,h_relu3,h_relu4,h_relu5,h_relu6,h_relu7)

        return out


class alexnet(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(alexnet, self).__init__()
        alexnet_pretrained_features = tv.alexnet(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(2):
            self.slice1.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(10, 12):
            self.slice5.add_module(str(x), alexnet_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        alexnet_outputs = namedtuple("AlexnetOutputs", ['relu1', 'relu2', 'relu3', 'relu4', 'relu5'])
        out = alexnet_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)

        return out


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = tv.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)

        return out


class resnet(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True, num=18):
        super(resnet, self).__init__()
        if(num==18):
            self.net = tv.resnet18(pretrained=pretrained)
        elif(num==34):
            self.net = tv.resnet34(pretrained=pretrained)
        elif(num==50):
            self.net = tv.resnet50(pretrained=pretrained)
        elif(num==101):
            self.net = tv.resnet101(pretrained=pretrained)
        elif(num==152):
            self.net = tv.resnet152(pretrained=pretrained)
        self.N_slices = 5

        self.conv1 = self.net.conv1
        self.bn1 = self.net.bn1
        self.relu = self.net.relu
        self.maxpool = self.net.maxpool
        self.layer1 = self.net.layer1
        self.layer2 = self.net.layer2
        self.layer3 = self.net.layer3
        self.layer4 = self.net.layer4

        # freeze the net
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.conv1(X)
        h = self.bn1(h)
        h = self.relu(h)
        h_relu1 = h
        h = self.maxpool(h)
        h = self.layer1(h)
        h_conv2 = h
        h = self.layer2(h)
        h_conv3 = h
        h = self.layer3(h)
        h_conv4 = h
        h = self.layer4(h)
        h_conv5 = h

        outputs = namedtuple("Outputs", ['relu1','conv2','conv3','conv4','conv5'])
        out = outputs(h_relu1, h_conv2, h_conv3, h_conv4, h_conv5)

        return out


class vgg19_any_layer(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True, indices=None):
        super(vgg19_any_layer, self).__init__()
        self.vgg_pretrained_features = tv.vgg19(pretrained=pretrained).features
        self.indices = indices
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        out = []
        for i in range(32):
            X = self.vgg_pretrained_features[i](X)
            if self.indices is None:
                out.append(X)
            elif (i+1) in self.indices:
                out.append(X)

        return out


class vgg16_any_layer(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True, indices=None):
        super(vgg16_any_layer, self).__init__()
        self.vgg_pretrained_features = tv.vgg16(pretrained=pretrained).features
        self.indices = indices
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        out = []
        for i in range(30):
            X = self.vgg_pretrained_features[i](X)
            if self.indices is None:
                out.append(X)
            elif (i+1) in self.indices:
                out.append(X)

        return out




## below codes are from the RCF net: https://github.com/yun-liu/RCF
# @article{liu2019richer,
#   title={Richer Convolutional Features for Edge Detection},
#   author={Liu, Yun and Cheng, Ming-Ming and Hu, Xiaowei and Bian, Jia-Wang and Zhang, Le and Bai, Xiang and Tang, Jinhui},
#   journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
#   volume={41},
#   number={8},
#   pages={1939--1946},
#   year={2019},
#   publisher={IEEE}
# }

import torch.nn as nn
import numpy as np

class RCF(nn.Module):
    def __init__(self, requires_grad=False):
        super(RCF, self).__init__()
        #lr 1 2 decay 1 0      
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3,
                        stride=1, padding=2, dilation=2)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3,
                        stride=1, padding=2, dilation=2)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3,
                        stride=1, padding=2, dilation=2)

        self.activ =  nn.ReLU(inplace=True)

        self.maxpool4 = nn.MaxPool2d(2, stride=1, ceil_mode=True)
        self.maxpool_1  =  nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.maxpool_2  =  nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.maxpool_3  =  nn.MaxPool2d(2, stride=2, ceil_mode=True)

        #lr 0.1 0.2 decay 1 0
        self.conv1_1_down = nn.Conv2d(64, 21, 1, padding=0)
        self.conv1_2_down = nn.Conv2d(64, 21, 1, padding=0)

        self.conv2_1_down = nn.Conv2d(128, 21, 1, padding=0)
        self.conv2_2_down = nn.Conv2d(128, 21, 1, padding=0)

        self.conv3_1_down = nn.Conv2d(256, 21, 1, padding=0)
        self.conv3_2_down = nn.Conv2d(256, 21, 1, padding=0)
        self.conv3_3_down = nn.Conv2d(256, 21, 1, padding=0)

        self.conv4_1_down = nn.Conv2d(512, 21, 1, padding=0)
        self.conv4_2_down = nn.Conv2d(512, 21, 1, padding=0)
        self.conv4_3_down = nn.Conv2d(512, 21, 1, padding=0)
        
        self.conv5_1_down = nn.Conv2d(512, 21, 1, padding=0)
        self.conv5_2_down = nn.Conv2d(512, 21, 1, padding=0)
        self.conv5_3_down = nn.Conv2d(512, 21, 1, padding=0)

        #lr 0.01 0.02 decay 1 0
        self.score_dsn1 = nn.Conv2d(21, 1, 1)
        self.score_dsn2 = nn.Conv2d(21, 1, 1)
        self.score_dsn3 = nn.Conv2d(21, 1, 1)
        self.score_dsn4 = nn.Conv2d(21, 1, 1)
        self.score_dsn5 = nn.Conv2d(21, 1, 1)
        #lr 0.001 0.002 decay 1 0
        self.score_final = nn.Conv2d(5, 1, 1)

        ## Fixed the upsampling weights for the training process as per @https://github.com/xwjabc/hed
        self.weight_deconv2 =  make_bilinear_weights(4, 1).cuda()
        self.weight_deconv3 =  make_bilinear_weights(8, 1).cuda()
        self.weight_deconv4 =  make_bilinear_weights(16, 1).cuda()
        # Wrong Deconv Filter size. Updated from RCF yun_liu
        # self.weight_deconv5 =  make_bilinear_weights(32, 1).cuda()
        self.weight_deconv5 =  make_bilinear_weights(16, 1).cuda()

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        # VGG
        img_H, img_W = x.shape[2], x.shape[3]
        conv1_1 = self.activ(self.conv1_1(x))
        conv1_2 = self.activ(self.conv1_2(conv1_1))
        pool1   = self.maxpool_1(conv1_2)

        conv2_1 = self.activ(self.conv2_1(pool1))
        conv2_2 = self.activ(self.conv2_2(conv2_1))
        pool2   = self.maxpool_2(conv2_2)

        conv3_1 = self.activ(self.conv3_1(pool2))
        conv3_2 = self.activ(self.conv3_2(conv3_1))
        conv3_3 = self.activ(self.conv3_3(conv3_2))
        pool3   = self.maxpool_3(conv3_3)

        conv4_1 = self.activ(self.conv4_1(pool3))
        conv4_2 = self.activ(self.conv4_2(conv4_1))
        conv4_3 = self.activ(self.conv4_3(conv4_2))
        pool4   = self.maxpool4(conv4_3)

        conv5_1 = self.activ(self.conv5_1(pool4))
        conv5_2 = self.activ(self.conv5_2(conv5_1))
        conv5_3 = self.activ(self.conv5_3(conv5_2))

        conv1_1_down = self.conv1_1_down(conv1_1)
        conv1_2_down = self.conv1_2_down(conv1_2)
        conv2_1_down = self.conv2_1_down(conv2_1)
        conv2_2_down = self.conv2_2_down(conv2_2)
        conv3_1_down = self.conv3_1_down(conv3_1)
        conv3_2_down = self.conv3_2_down(conv3_2)
        conv3_3_down = self.conv3_3_down(conv3_3)
        conv4_1_down = self.conv4_1_down(conv4_1)
        conv4_2_down = self.conv4_2_down(conv4_2)
        conv4_3_down = self.conv4_3_down(conv4_3)
        conv5_1_down = self.conv5_1_down(conv5_1)
        conv5_2_down = self.conv5_2_down(conv5_2)
        conv5_3_down = self.conv5_3_down(conv5_3)

        so1_out = self.score_dsn1(conv1_1_down + conv1_2_down)
        so2_out = self.score_dsn2(conv2_1_down + conv2_2_down)
        so3_out = self.score_dsn3(conv3_1_down + conv3_2_down + conv3_3_down)
        so4_out = self.score_dsn4(conv4_1_down + conv4_2_down + conv4_3_down)
        so5_out = self.score_dsn5(conv5_1_down + conv5_2_down + conv5_3_down)

        upsample2 = torch.nn.functional.conv_transpose2d(so2_out, self.weight_deconv2, stride=2)
        upsample3 = torch.nn.functional.conv_transpose2d(so3_out, self.weight_deconv3, stride=4)
        upsample4 = torch.nn.functional.conv_transpose2d(so4_out, self.weight_deconv4, stride=8)
        upsample5 = torch.nn.functional.conv_transpose2d(so5_out, self.weight_deconv5, stride=8)
        
        ### center crop
        so1 = crop(so1_out, img_H, img_W, 0 , 0)
        so2 = crop(upsample2, img_H, img_W , 1, 1 )
        so3 = crop(upsample3, img_H, img_W , 2, 2 )
        so4 = crop(upsample4, img_H, img_W , 4, 4)
        so5 = crop(upsample5, img_H, img_W , 0, 0)

        fusecat = torch.cat((so1, so2, so3, so4, so5), dim=1)
        fuse = self.score_final(fusecat)
        results = [so1, so2, so3, so4, so5, fuse]
        results = [torch.sigmoid(r) for r in results]
        # return results
        out = [conv1_2, conv2_2, conv3_3, conv4_3, conv5_3]
        return out

# Based on BDCN Implementation @ https://github.com/pkuCactus/BDCN
def crop(data1, h, w , crop_h, crop_w):
    _, _, h1, w1 = data1.size()
    assert(h <= h1 and w <= w1)
    data = data1[:, :, crop_h:crop_h+h, crop_w:crop_w+w]
    return data

def make_bilinear_weights(size, num_channels):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    # print(filt)
    filt = torch.from_numpy(filt)
    w = torch.zeros(num_channels, num_channels, size, size)
    w.requires_grad = False
    for i in range(num_channels):
        for j in range(num_channels):
            if i == j:
                w[i, j] = filt
    return w

def upsample(input, stride, num_channels=1):
    kernel_size = stride * 2
    kernel = make_bilinear_weights(kernel_size, num_channels).cuda()
    return torch.nn.functional.conv_transpose2d(input, kernel, stride=stride)