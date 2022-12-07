## ------------------------------------------------------------------------------------------
## Confusing image quality assessment: Towards better augmented reality experience
## Huiyu Duan, Xiongkuo Min, Yucheng Zhu, Guangtao Zhai, Xiaokang Yang, and Patrick Le Callet
## IEEE Transactions on Image Processing (TIP)
## ------------------------------------------------------------------------------------------

from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from . import pretrained_networks as pn
import torch.nn

import lpips

import scipy.io as sio

def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2,3],keepdim=keepdim)

def spatial_sal(in_tens, sal, keepdim=True):
    sal = nn.functional.interpolate(sal, size=(in_tens.shape[2], in_tens.shape[3]), mode='bicubic', align_corners=False)
    return (in_tens*sal).sum([2,3],keepdim=keepdim)/sal.sum([2,3],keepdim=keepdim)

def upsample(in_tens, out_HW=(64,64)): # assumes scale factor is same for H and W
    in_H, in_W = in_tens.shape[2], in_tens.shape[3]
    return nn.Upsample(size=out_HW, mode='bilinear', align_corners=False)(in_tens)

# Learned perceptual metric
class LPIPS(nn.Module):
    def __init__(self, pretrained=True, net='alex', version='0.1', lpips=True, spatial=False, 
        pnet_rand=False, pnet_tune=False, use_dropout=True, model_path=None, eval_mode=True, verbose=True):
        # lpips - [True] means with linear calibration on top of base network
        # pretrained - [True] means load linear weights

        super(LPIPS, self).__init__()
        if(verbose):
            print('Setting up [%s] perceptual loss: trunk [%s], v[%s], spatial [%s]'%
                ('LPIPS' if lpips else 'baseline', net, version, 'on' if spatial else 'off'))

        self.pnet_type = net
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.spatial = spatial
        self.lpips = lpips # false means baseline of just averaging all layers
        self.version = version
        self.scaling_layer = ScalingLayer()

        if(self.pnet_type in ['vgg','vgg16']):
            net_type = pn.vgg16
            self.chns = [64,128,256,512,512]
        elif(self.pnet_type=='alex'):
            net_type = pn.alexnet
            self.chns = [64,192,384,256,256]
        elif(self.pnet_type=='squeeze'):
            net_type = pn.squeezenet
            self.chns = [64,128,256,384,384,512,512]
        elif(self.pnet_type=='resnet18'):
            net_type = pn.resnet
            self.chns = [64,64,128,256,512]
        elif(self.pnet_type=='resnet34'):
            net_type = pn.resnet
            self.chns = [64,64,128,256,512]
        elif(self.pnet_type=='resnet50'):
            net_type = pn.resnet
            self.chns = [64,256,512,1024,2048]
        elif(self.pnet_type=='vgg19'):
            net_type = pn.vgg19_any_layer
            indices = [2, 7, 12, 21, 30]
            self.chns = [64,128,256,512,512]
        elif(self.pnet_type=='vgg19_all_layers'):
            net_type = pn.vgg19_any_layer
            indices = None
            self.chns = [64,128,256,512,512]
        elif(self.pnet_type=='vgg16_plus'):
            net_type = pn.vgg16_any_layer
            # indices = [4, 9, 16, 23, 30]
            indices = [7, 23, 26, 28, 30]
            # indices = [2, 7, 14, 23, 26]
            self.chns = [64,128,256,512,512]
            # self.chns = [128,512,512,512,512]
        elif(self.pnet_type=='vgg16_all_layers'):
            net_type = pn.vgg16_any_layer
            indices = None
            self.chns = [64,128,256,512,512]
        self.L = len(self.chns)

        if(self.pnet_type=='resnet18'):
            self.net = net_type(pretrained=not self.pnet_rand, requires_grad=self.pnet_tune, num=18)
        elif(self.pnet_type=='resnet34'):
            self.net = net_type(pretrained=not self.pnet_rand, requires_grad=self.pnet_tune, num=34)
        elif(self.pnet_type=='resnet50'):
            self.net = net_type(pretrained=not self.pnet_rand, requires_grad=self.pnet_tune, num=50)
        elif(self.pnet_type in ['vgg19','vgg19_all_layers','vgg16_plus','vgg16_all_layers']):
            self.net = net_type(pretrained=not self.pnet_rand, requires_grad=self.pnet_tune, indices=indices)
        else:
            self.net = net_type(pretrained=not self.pnet_rand, requires_grad=self.pnet_tune)

        if(lpips):
            self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
            self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
            self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
            self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
            self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
            self.lins = [self.lin0,self.lin1,self.lin2,self.lin3,self.lin4]
            if(self.pnet_type=='squeeze'): # 7 layers for squeezenet
                self.lin5 = NetLinLayer(self.chns[5], use_dropout=use_dropout)
                self.lin6 = NetLinLayer(self.chns[6], use_dropout=use_dropout)
                self.lins+=[self.lin5,self.lin6]
            self.lins = nn.ModuleList(self.lins)

            if(pretrained):
                if(model_path is None):
                    import inspect
                    import os
                    model_path = os.path.abspath(os.path.join(inspect.getfile(self.__init__), '..', 'weights/v%s/%s.pth'%(version,net)))

                if(verbose):
                    print('Loading model from: %s'%model_path)
                self.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)          
        
        self.edge = True
        if(self.edge):
            self.lin_e0 = NetLinLayer(64, use_dropout=use_dropout)
            self.lin_e1 = NetLinLayer(128, use_dropout=use_dropout)
            self.lin_e2 = NetLinLayer(256, use_dropout=use_dropout)
            self.lin_e3 = NetLinLayer(512, use_dropout=use_dropout)
            self.lin_e4 = NetLinLayer(512, use_dropout=use_dropout)
            self.lins_e = [self.lin_e0,self.lin_e1,self.lin_e2,self.lin_e3,self.lin_e4]
            if(self.pnet_type=='squeeze'): # 7 layers for squeezenet
                self.lin_e5 = NetLinLayer(self.chns[5], use_dropout=use_dropout)
                self.lin_e6 = NetLinLayer(self.chns[6], use_dropout=use_dropout)
                self.lins_e+=[self.lin_e5,self.lin_e6]
            self.lins_e = nn.ModuleList(self.lins_e)
            
            self.net_e = pn.RCF()
            vgg16 = sio.loadmat('./lpips/weights/vgg16convs.mat')
            torch_params =  self.net_e.state_dict()
            for k in vgg16.keys():
                name_par = k.split('-')
                size = len(name_par)
                if size  == 2:
                    name_space = name_par[0] + '.' + name_par[1]            
                    data = np.squeeze(vgg16[k])
                    torch_params[name_space] = torch.from_numpy(data)    
            self.net_e.load_state_dict(torch_params)

        if(eval_mode):
            self.eval()

    def forward(self, in0, in1, sal=None, retPerLayer=False, normalize=False):
        if normalize: # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            in0 = 2 * in0  - 1
            in1 = 2 * in1  - 1

        # v0.0 - original release had a bug, where input was not scaled
        in0_input, in1_input = (self.scaling_layer(in0), self.scaling_layer(in1)) if self.version=='0.1' else (in0, in1)
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        self.L = len(outs0)

        for kk in range(self.L):
            feats0[kk], feats1[kk] = lpips.normalize_tensor(outs0[kk]), lpips.normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk]-feats1[kk])**2  # 0: ref, 1: dis

        if(self.lpips):
            if(self.spatial):
                res = [upsample(self.lins[kk](diffs[kk]), out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                if sal is not None:
                    res = [spatial_sal(self.lins[kk](diffs[kk]), sal, keepdim=True) for kk in range(self.L)]
                else:
                    res = [spatial_average(self.lins[kk](diffs[kk]), keepdim=True) for kk in range(self.L)]
        else:
            if(self.spatial):
                res = [upsample(diffs[kk].sum(dim=1,keepdim=True), out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [spatial_average(diffs[kk].sum(dim=1,keepdim=True), keepdim=True) for kk in range(self.L)]

        if(self.edge):
            outs0_e, outs1_e = self.net_e.forward(in0_input), self.net_e.forward(in1_input)
            feats0_e, feats1_e, diffs_e = {}, {}, {}

            self.L = len(outs0_e)

            for kk in range(self.L):
                feats0_e[kk], feats1_e[kk] = lpips.normalize_tensor(outs0_e[kk]), lpips.normalize_tensor(outs1_e[kk])
                diffs_e[kk] = (feats0_e[kk]-feats1_e[kk])**2
            
            if sal is not None:
                res += [spatial_sal(self.lins_e[kk](diffs_e[kk]), sal, keepdim=True) for kk in range(self.L)]
            else:
                res += [spatial_average(self.lins_e[kk](diffs_e[kk]), keepdim=True) for kk in range(self.L)]

        # val = res[0]
        # for l in range(1,self.L):
        #     val += res[l]

        val = 0
        for l in range(self.L):
            val += res[l]

        if(self.edge):
            for l in range(self.L,self.L*2):
                val += res[l]
        
        if(retPerLayer):
            return (val, res)
        else:
            return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030,-.088,-.188])[None,:,None,None])
        self.register_buffer('scale', torch.Tensor([.458,.448,.450])[None,:,None,None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = [nn.Dropout(),] if(use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),]
        self.model = nn.Sequential(*layers)
        self.se_layer = SELayer(chn_in, reduction=int(chn_in/32))

    def forward(self, x):
        x = self.se_layer(x)
        return self.model(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if channel < reduction:
            reduction = channel
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class Dist2LogitLayer(nn.Module):
    ''' takes 2 distances, puts through fc layers, spits out value between [0,1] (if use_sigmoid is True) '''
    def __init__(self, chn_mid=32, use_sigmoid=True):
        super(Dist2LogitLayer, self).__init__()

        layers = [nn.Conv2d(5, chn_mid, 1, stride=1, padding=0, bias=True),]
        layers += [nn.LeakyReLU(0.2,True),]
        layers += [nn.Conv2d(chn_mid, chn_mid, 1, stride=1, padding=0, bias=True),]
        layers += [nn.LeakyReLU(0.2,True),]
        layers += [nn.Conv2d(chn_mid, 1, 1, stride=1, padding=0, bias=True),]
        if(use_sigmoid):
            layers += [nn.Sigmoid(),]
        self.model = nn.Sequential(*layers)

    def forward(self,d0,d1,eps=0.1):
        return self.model.forward(torch.cat((d0,d1,d0-d1,d0/(d1+eps),d1/(d0+eps)),dim=1))

class BCERankingLoss(nn.Module):
    def __init__(self, chn_mid=32):
        super(BCERankingLoss, self).__init__()
        self.net = Dist2LogitLayer(chn_mid=chn_mid)
        # self.parameters = list(self.net.parameters())
        self.loss = torch.nn.BCELoss()

    def forward(self, d0, d1, judge):
        per = (judge+1.)/2.
        self.logit = self.net.forward(d0,d1)
        return self.loss(self.logit, per)


class BCEScoringLoss(nn.Module):
    def __init__(self):
        super(BCEScoringLoss, self).__init__()
        layers = [nn.Conv2d(1, 1, 1, stride=1, padding=0, bias=True),]
        layers += [nn.LeakyReLU(0.2,True),]
        layers += [nn.Conv2d(1, 1, 1, stride=1, padding=0, bias=True),]
        layers += [nn.Sigmoid(),]
        self.net = nn.Sequential(*layers)
        # self.parameters = list(self.net.parameters())
        self.loss = torch.nn.BCELoss()

    def compute(self,d):
        return self.net.forward(d)

    def forward(self, d, y):
        self.logit = self.net.forward(d)
        # self.logit = d
        return self.loss(self.logit, y)


# L2, DSSIM metrics
class FakeNet(nn.Module):
    def __init__(self, use_gpu=True, colorspace='Lab'):
        super(FakeNet, self).__init__()
        self.use_gpu = use_gpu
        self.colorspace = colorspace


class L2(FakeNet):
    def forward(self, in0, in1, retPerLayer=None):
        assert(in0.size()[0]==1) # currently only supports batchSize 1

        if(self.colorspace=='RGB'):
            (N,C,X,Y) = in0.size()
            value = torch.mean(torch.mean(torch.mean((in0-in1)**2,dim=1).view(N,1,X,Y),dim=2).view(N,1,1,Y),dim=3).view(N)
            return value
        elif(self.colorspace=='Lab'):
            value = lpips.l2(lpips.tensor2np(lpips.tensor2tensorlab(in0.data,to_norm=False)), 
                lpips.tensor2np(lpips.tensor2tensorlab(in1.data,to_norm=False)), range=100.).astype('float')
            ret_var = Variable( torch.Tensor((value,) ) )
            if(self.use_gpu):
                ret_var = ret_var.cuda()
            return ret_var


class DSSIM(FakeNet):

    def forward(self, in0, in1, retPerLayer=None):
        assert(in0.size()[0]==1) # currently only supports batchSize 1

        if(self.colorspace=='RGB'):
            value = lpips.dssim(1.*lpips.tensor2im(in0.data), 1.*lpips.tensor2im(in1.data), range=255.).astype('float')
        elif(self.colorspace=='Lab'):
            value = lpips.dssim(lpips.tensor2np(lpips.tensor2tensorlab(in0.data,to_norm=False)), 
                lpips.tensor2np(lpips.tensor2tensorlab(in1.data,to_norm=False)), range=100.).astype('float')
        ret_var = Variable( torch.Tensor((value,) ) )
        if(self.use_gpu):
            ret_var = ret_var.cuda()
        return ret_var


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Network',net)
    print('Total number of parameters: %d' % num_params)
