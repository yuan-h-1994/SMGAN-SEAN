import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm2d, LeakyReLU, ConvTranspose2d, ReLU, Tanh, InstanceNorm2d, ReplicationPad2d
import torch.nn.functional as F
import random
from utils import gaussian, to_var, to_data, save_image, label_input, ACE
from vgg import GramMSELoss, SemanticFeature
import numpy as np
import math
import torch.autograd as autograd
from torch.autograd import Variable
import os
import torch.nn.utils.spectral_norm as spectral_norm
from options import TrainShapeMatchingOptions
from options import TestOptions
import re

id = 0 # for saving network output to file during training

#######################  Texture Network
# based on Convolution-BatchNorm-ReLU
class myTConv(nn.Module):
    def __init__(self, num_filter=128, stride=1, in_channels=128):
        super(myTConv, self).__init__()
        
        self.pad = ReplicationPad2d(padding=1)
        self.conv = Conv2d(out_channels=num_filter, kernel_size=3, 
                           stride=stride, padding=0, in_channels=in_channels)
        self.bn = BatchNorm2d(num_features=num_filter, track_running_stats=True)
        self.relu = ReLU()
            
    def forward(self, x):
        return self.relu(self.bn(self.conv(self.pad(x))))

class myTBlock(nn.Module):
    def __init__(self, num_filter=128, p=0.0):
        super(myTBlock, self).__init__()
        
        self.myconv = myTConv(num_filter=num_filter, stride=1, in_channels=128)
        self.pad = ReplicationPad2d(padding=1)
        self.conv = Conv2d(out_channels=num_filter, kernel_size=3, padding=0, in_channels=128)
        self.bn = BatchNorm2d(num_features=num_filter, track_running_stats=True)
        self.relu = ReLU()
        self.dropout = nn.Dropout(p=p)
        
    def forward(self, x):
        return self.dropout(self.relu(x+self.bn(self.conv(self.pad(self.myconv(x))))))

class TextureGenerator(nn.Module):
    def __init__(self, ngf = 32, n_layers = 5):
        super(TextureGenerator, self).__init__()
        #our_norm_type = 'spadesyncbatch3x3'

        modelList1 = []
        modelList2 = []
        modelList3 = []
        modelList4 = []
        modelList1.append(ReplicationPad2d(padding=4))
        modelList1.append(Conv2d(out_channels=ngf, kernel_size=9, padding=0, in_channels=3+3))
        modelList1.append(ReLU())
        modelList1.append(myTConv(ngf*2, 2, ngf))
        modelList1.append(myTConv(ngf*4, 2, ngf*2))
        
        for n in range(int(n_layers/2)): 
            modelList1.append(myTBlock(ngf*4, p=0.0))
        # dropout to make model more robust
        modelList1.append(myTBlock(ngf*4, p=0.5))
        for n in range(int(n_layers/2)+1,n_layers):
            modelList1.append(myTBlock(ngf*4, p=0.0))  
        
        modelList1.append(ConvTranspose2d(out_channels=ngf*2, kernel_size=4, stride=2, padding=0, in_channels=ngf*4))
        #modelList.append(BatchNorm2d(num_features=ngf*2, track_running_stats=True))
        modelList2.append(ReLU())
        modelList2.append(ConvTranspose2d(out_channels=ngf, kernel_size=4, stride=2, padding=0, in_channels=ngf*2))
        #modelList.append(BatchNorm2d(num_features=ngf, track_running_stats=True))
        modelList3.append(ReLU())
        modelList3.append(ReplicationPad2d(padding=1))
        modelList3.append(Conv2d(out_channels=3, kernel_size=9, padding=0, in_channels=ngf))
        modelList4.append(Tanh())

        self.head_0 = SPADEResnetBlock(ngf*2, ngf*2)
        self.head_1 = SPADEResnetBlock(ngf, ngf)
        self.head_2 = SPADEResnetBlock(3, 3)
        self.model1 = nn.Sequential(*modelList1)
        self.model2 = nn.Sequential(*modelList2)
        self.model3 = nn.Sequential(*modelList3)
        self.model4 = nn.Sequential(*modelList4)
        #self.ace_0 = ACE(our_norm_type, norm_nc=3, label_nc=3)

    def forward(self, x, labels,style_codem):
        #print('x',x.size())
        #print('labels',labels.size())
        #gamma_avg, beta_avg = self.ace_0(x, labels, style_codem)
        x = torch.cat([x, labels], dim = 1)
        out1 = self.model1(x)
        out2 = self.head_0(out1,labels,style_codem)
        out3 = self.model2(out2)
        out4 = self.head_1(out3,labels,style_codem)
        out5 = self.model3(out4)
        out6 = self.head_2(out5,labels,style_codem)
        out7 = self.model4(out6)
        return out7
        
###################### Glyph Network    
# based on Convolution-BatchNorm-LeakyReLU    
class myGConv(nn.Module):
    def __init__(self, num_filter=128, stride=1, in_channels=128):
        super(myGConv, self).__init__()
        self.pad = ReplicationPad2d(padding=1)
        self.conv = Conv2d(out_channels=num_filter, kernel_size=3, 
                           stride=stride, padding=0, in_channels=in_channels)
        self.bn = BatchNorm2d(num_features=num_filter, track_running_stats=True)
        # either ReLU or LeakyReLU is OK
        self.relu = LeakyReLU(0.2)
            
    def forward(self, x):
        return self.relu(self.bn(self.conv(self.pad(x))))

class myGBlock(nn.Module):
    def __init__(self, num_filter=128):
        super(myGBlock, self).__init__()
        
        self.myconv = myGConv(num_filter=num_filter, stride=1, in_channels=num_filter)
        self.pad = ReplicationPad2d(padding=1)
        self.conv = Conv2d(out_channels=num_filter, kernel_size=3, padding=0, in_channels=num_filter)
        self.bn = BatchNorm2d(num_features=num_filter, track_running_stats=True)
        
    def forward(self, x):
        return x+self.bn(self.conv(self.pad(self.myconv(x))))

# Controllable ResBlock
class myGCombineBlock(nn.Module):
    def __init__(self, num_filter=128, p=0.0):
        super(myGCombineBlock, self).__init__()
        
        self.myBlock1 = myGBlock(num_filter=num_filter)
        self.myBlock2 = myGBlock(num_filter=num_filter)
        self.relu = LeakyReLU(0.2)
        self.label = 1.0
        self.dropout = nn.Dropout(p=p)
        
    def myCopy(self):
        self.myBlock1.load_state_dict(self.myBlock2.state_dict())
        
    def forward(self, x):
        return self.dropout(self.relu(self.myBlock1(x)*self.label + self.myBlock2(x)*(1.0-self.label)))
    
class GlyphGenerator(nn.Module):
    def __init__(self, ngf=32, n_layers = 5):
        super(GlyphGenerator, self).__init__()
        
        encoder = []
        encoder.append(ReplicationPad2d(padding=4))
        encoder.append(Conv2d(out_channels=ngf, kernel_size=9, padding=0, in_channels=3+3))
        encoder.append(LeakyReLU(0.2))
        encoder.append(myGConv(ngf*2, 2, ngf))
        encoder.append(myGConv(ngf*4, 2, ngf*2))

        transformer = []
        for n in range(int(n_layers/2)-1):
            transformer.append(myGCombineBlock(ngf*4,p=0.0))
        # dropout to make model more robust    
        transformer.append(myGCombineBlock(ngf*4,p=0.5))
        transformer.append(myGCombineBlock(ngf*4,p=0.5))
        for n in range(int(n_layers/2)+1,n_layers):
            transformer.append(myGCombineBlock(ngf*4,p=0.0))  
        
        decoder1 = []
        decoder1.append(ConvTranspose2d(out_channels=ngf*2, kernel_size=4, stride=2, padding=0, in_channels=ngf*4))
        #decoder.append(BatchNorm2d(num_features=ngf*2, track_running_stats=True))
        decoder3 = []
        decoder3.append(LeakyReLU(0.2))
        decoder3.append(ConvTranspose2d(out_channels=ngf, kernel_size=4, stride=2, padding=0, in_channels=ngf*2))
        #decoder.append(BatchNorm2d(num_features=ngf, track_running_stats=True))
        decoder5 = []
        decoder5.append(LeakyReLU(0.2))
        decoder5.append(ReplicationPad2d(padding=1))
        decoder5.append(Conv2d(out_channels=3, kernel_size=9, padding=0, in_channels=ngf))
        decoder6 = []
        decoder6.append(Tanh())
        
        self.head_0 = SPADEResnetBlock(ngf*2, ngf*2)
        self.head_1 = SPADEResnetBlock(ngf, ngf)
        self.head_2 = SPADEResnetBlock(3, 3)
        self.encoder = nn.Sequential(*encoder)
        self.transformer = nn.Sequential(*transformer)
        self.decoder1 = nn.Sequential(*decoder1)
        self.decoder3 = nn.Sequential(*decoder3)
        self.decoder6 = nn.Sequential(*decoder6)
        self.decoder5 = nn.Sequential(*decoder5)
    
    def myCopy(self):
        for myCombineBlock in self.transformer:
            myCombineBlock.myCopy()
            
    # controlled by Controllable ResBlcok    
    def forward(self, x, l, labels,style_codem):
        for myCombineBlock in self.transformer:
            # label smoothing [-1,1]-->[0.9,0.1]
            myCombineBlock.label = (1.0-l)*0.4+0.1
        #labels = label_input(opts.label_name,opts.gpu)
        #labels = to_var(labels) 
        #l_img = l.expand(l.size(0), l.size(1), x.size(2), x.size(3))
        #print('label',labels.size())
        x = torch.cat([x, labels], dim = 1)
        #print('xlab',x.size())
        #out0 = self.encoder(torch.cat([x, labels], dim = 1))
        out0 = self.encoder(x)
        out1 = self.transformer(out0)
        out2 = self.decoder1(out1)
        out3 = self.head_0(out2,labels,style_codem)
        out4 = self.decoder3(out3)
        out5 = self.head_1(out4,labels,style_codem)
        out6 = self.decoder5(out5)
        out7 = self.head_2(out6,labels,style_codem)
        out8 = self.decoder6(out7)
        return out8

    
##################### Sketch Module 
# based on Convolution-InstanceNorm-ReLU   
# Smoothness Block
class myBlur(nn.Module):
    def __init__(self, kernel_size=121, channels=3):
        super(myBlur, self).__init__()
        kernel_size = int(int(kernel_size/2)*2)+1
        self.kernel_size=kernel_size
        self.channels = channels
        self.GF = nn.Conv2d(in_channels=channels, out_channels=channels,
                                    kernel_size=kernel_size, groups=channels, bias=False)
        x_cord = torch.arange(self.kernel_size+0.)
        x_grid = x_cord.repeat(self.kernel_size).view(self.kernel_size, self.kernel_size)
        y_grid = x_grid.t()
        self.xy_grid = torch.stack([x_grid, y_grid], dim=-1)
        self.mean = (self.kernel_size - 1)//2
        self.diff = -torch.sum((self.xy_grid - self.mean)**2., dim=-1)
        self.gaussian_filter = nn.Conv2d(in_channels=self.channels, out_channels=self.channels,
                                    kernel_size=self.kernel_size, groups=self.channels, bias=False)

        self.gaussian_filter.weight.requires_grad = False
        
    def forward(self, x, sigma, gpu):
        sigma = sigma * 8. + 16.
        variance = sigma**2.
        gaussian_kernel = (1./(2.*math.pi*variance)) * torch.exp(self.diff /(2*variance))
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        gaussian_kernel = gaussian_kernel.view(1, 1, self.kernel_size, self.kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(self.channels, 1, 1, 1)
        if gpu:
            gaussian_kernel = gaussian_kernel.cuda()
        self.gaussian_filter.weight.data = gaussian_kernel
        return self.gaussian_filter(F.pad(x, (self.mean,self.mean,self.mean,self.mean), "replicate")) 

class mySConv(nn.Module):
    def __init__(self, num_filter=128, stride=1, in_channels=128):
        super(mySConv, self).__init__()
        
        self.conv = Conv2d(out_channels=num_filter, kernel_size=3, 
                           stride=stride, padding=1, in_channels=in_channels)
        self.bn = InstanceNorm2d(num_features=num_filter)
        self.relu = ReLU()
            
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class mySBlock(nn.Module):
    def __init__(self, num_filter=128):
        super(mySBlock, self).__init__()
        
        self.myconv = mySConv(num_filter=num_filter, stride=1, in_channels=num_filter)
        self.conv = Conv2d(out_channels=num_filter, kernel_size=3, padding=1, in_channels=num_filter)
        self.bn = InstanceNorm2d(num_features=num_filter)
        self.relu = ReLU()
        
    def forward(self, x):
        return self.relu(x+self.bn(self.conv(self.myconv(x))))
    
# Transformation Block
class SketchGenerator(nn.Module):
    def __init__(self, in_channels = 4, ngf = 32, n_layers = 5):
        super(SketchGenerator, self).__init__()
        
        encoder = []
        encoder.append(Conv2d(out_channels=ngf, kernel_size=9, padding=4, in_channels=in_channels))
        encoder.append(ReLU())
        encoder.append(mySConv(ngf*2, 2, ngf))
        encoder.append(mySConv(ngf*4, 2, ngf*2))
        
        transformer = []
        for n in range(n_layers):
            transformer.append(mySBlock(ngf*4+1))
        
        decoder1 = []
        decoder2 = []
        decoder3 = []
        decoder1.append(ConvTranspose2d(out_channels=ngf*2, kernel_size=4, stride=2, padding=0, in_channels=ngf*4+2))
        decoder1.append(InstanceNorm2d(num_features=ngf*2))
        decoder1.append(ReLU())
        decoder2.append(ConvTranspose2d(out_channels=ngf, kernel_size=4, stride=2, padding=0, in_channels=ngf*2+1))
        decoder2.append(InstanceNorm2d(num_features=ngf))
        decoder2.append(ReLU())
        decoder3.append(Conv2d(out_channels=3, kernel_size=9, padding=1, in_channels=ngf+1))
        decoder3.append(Tanh())
        
        self.encoder = nn.Sequential(*encoder)
        self.transformer = nn.Sequential(*transformer)
        self.decoder1 = nn.Sequential(*decoder1)
        self.decoder2 = nn.Sequential(*decoder2)
        self.decoder3 = nn.Sequential(*decoder3)
    
    # controlled by label concatenation
    def forward(self, x, l):
        l_img = l.expand(l.size(0), l.size(1), x.size(2), x.size(3))
        out0 = self.encoder(torch.cat([x, l_img], 1))
        l_img0 = l.expand(l.size(0), l.size(1), out0.size(2), out0.size(3))
        out1 = self.transformer(torch.cat([out0, l_img0], 1))
        l_img1 = l.expand(l.size(0), l.size(1), out1.size(2), out1.size(3))
        out2 = self.decoder1(torch.cat([out1, l_img1], 1))
        l_img2 = l.expand(l.size(0), l.size(1), out2.size(2), out2.size(3))
        out3 = self.decoder2(torch.cat([out2, l_img2], 1))
        l_img3 = l.expand(l.size(0), l.size(1), out3.size(2), out3.size(3))
        out4 = self.decoder3(torch.cat([out3, l_img3], 1))
        return out4


################ Discriminator_gtf
# for Texture Networks
# Resnet
class ResBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super().__init__()
        channel = channel_out // 4

        self.conv1 = nn.Conv2d(channel_in, channel,kernel_size=(1, 1))
        self.bn1 = nn.BatchNorm2d(channel)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(channel, channel,kernel_size=(3, 3),padding=1)
        self.bn2 = nn.BatchNorm2d(channel)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(channel, channel_out,kernel_size=(1, 1),padding=0)
        self.bn3 = nn.BatchNorm2d(channel_out)
        self.shortcut = self._shortcut(channel_in, channel_out)
        self.relu3 = nn.ReLU()
    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu1(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.relu2(h)
        h = self.conv3(h)
        h = self.bn3(h)
        shortcut = self.shortcut(x)
        y = self.relu3(h + shortcut)  # skip connection
        #print(y.size())
        return y
    def _shortcut(self, channel_in, channel_out):
        if channel_in != channel_out:
            return self._projection(channel_in, channel_out)
        else:
            return lambda x: x
    def _projection(self, channel_in, channel_out):
        return nn.Conv2d(channel_in, channel_out,
                         kernel_size=(1, 1),
                         padding=0)

class Discriminator_gtf(nn.Module):
    def __init__(self, in_channels, ngf = 32):
        super(Discriminator_gtf, self).__init__()
        encoder = []
        encoder.append(Conv2d(out_channels=ngf, kernel_size=9, stride=2,padding=4, in_channels=in_channels))
        encoder.append(ReLU())

        encoder.append(ResBlock(ngf, ngf))
        encoder.append(Conv2d(out_channels=ngf*2, kernel_size=3,stride=2, padding=1, in_channels=ngf))
        encoder.append(ReLU())

        encoder.append(ResBlock(ngf*2, ngf*2))
        encoder.append(Conv2d(out_channels=ngf*4, kernel_size=3,stride=2, padding=1, in_channels=ngf*2))
        encoder.append(ReLU())

        encoder.append(ResBlock(ngf*4, ngf*4))
        encoder.append(Conv2d(out_channels=ngf*8, kernel_size=3,stride=2, padding=1, in_channels=ngf*4))
        encoder.append(ReLU())

        encoder.append(ResBlock(ngf*8, ngf*8))
        encoder.append(Conv2d(out_channels=ngf*12, kernel_size=3,stride=2, padding=1, in_channels=ngf*8))
        encoder.append(ReLU())

        encoder.append(ResBlock(ngf*12, ngf*12))
        encoder.append(Conv2d(out_channels=ngf*12, kernel_size=3,stride=2, padding=1, in_channels=ngf*12))
        encoder.append(ReLU())

        encoder.append(Conv2d(out_channels=ngf*24, kernel_size=3, padding=1, in_channels=ngf*12))
        encoder.append(ReLU())
        encoder.append(Conv2d(out_channels=ngf*12, kernel_size=2, in_channels=ngf*24))
        encoder.append(ReLU())

        decoder1 = []
        decoder1.append(Linear(768,2048))
        decoder1.append(Linear(2048,2048))
        decoder1.append(Linear(2048,1024))
        decoder1.append(Linear(1024,1))

        self.encoder = nn.Sequential(*encoder)
        self.decoder1 = nn.Sequential(*decoder1)

    def forward(self, x1, x2):
        y1 = self.encoder(x1)
        y2 = self.encoder(x2)
        out1 = torch.cat([y1, y2], 1)
        out1 = out1.view(out1.size(0),-1)
        out2 = self.decoder1(out1)
        return out2
################ Discriminators
# Glyph and Texture Networks: BN
# Sketch Module: IN, multilayer
class Discriminator(nn.Module):
    def __init__(self, in_channels, ndf=32, n_layers=3, multilayer=False, IN=False):
        super(Discriminator, self).__init__()
        
        modelList = []    
        outlist1 = []
        outlist2 = []
        kernel_size = 4
        padding = int(np.ceil((kernel_size - 1)/2))
        modelList.append(Conv2d(out_channels=ndf, kernel_size=kernel_size, stride=2,
                              padding=2, in_channels=in_channels))
        modelList.append(LeakyReLU(0.2))

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 4)
            modelList.append(Conv2d(out_channels=ndf * nf_mult, kernel_size=kernel_size, stride=2,
                                  padding=2, in_channels=ndf * nf_mult_prev))
            if IN:
                modelList.append(InstanceNorm2d(num_features=ndf * nf_mult))
            else:
                modelList.append(BatchNorm2d(num_features=ndf * nf_mult, track_running_stats=True))
            modelList.append(LeakyReLU(0.2))

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 4)
        outlist1.append(Conv2d(out_channels=1, kernel_size=kernel_size, stride=1,
                              padding=padding, in_channels=ndf * nf_mult_prev))
        
        outlist2.append(Conv2d(out_channels=ndf * nf_mult, kernel_size=kernel_size, stride=1,
                              padding=padding, in_channels=ndf * nf_mult_prev))
        if IN:
            outlist2.append(InstanceNorm2d(num_features=ndf * nf_mult))
        else:
            outlist2.append(BatchNorm2d(num_features=ndf * nf_mult, track_running_stats=True))
        outlist2.append(LeakyReLU(0.2))
        outlist2.append(Conv2d(out_channels=1, kernel_size=kernel_size, stride=1,
                              padding=padding, in_channels=ndf * nf_mult))
        self.model = nn.Sequential(*modelList)
        self.out1 = nn.Sequential(*outlist1)
        self.out2 = nn.Sequential(*outlist2)
        self.multilayer = multilayer
        
    def forward(self, x):
        y = self.model(x)
        out2 = self.out2(y)
        if self.multilayer:
            out1 = self.out1(y)
            return torch.cat((out1.view(-1), out2.view(-1)), dim=0)
        else:
            return out2.view(-1)

        
######################## Sketch Module
class SketchModule(nn.Module):
    def __init__(self, G_layers = 6, D_layers = 5, ngf = 32, ndf = 32, gpu=True):
        super(SketchModule, self).__init__()
        
        self.G_layers = G_layers
        self.D_layers = D_layers
        self.ngf = ngf
        self.ndf = ndf
        self.gpu = gpu
        self.lambda_l1 = 100
        self.lambda_gp = 10
        self.lambda_adv = 1
        self.loss = nn.L1Loss()
        
        # Sketch Module = transformationBlock + smoothnessBlock
        # transformationBlock
        self.transBlock = SketchGenerator(4, self.ngf, self.G_layers)
        self.D_B = Discriminator(7, self.ndf, self.D_layers, True, True)
        # smoothnessBlock
        self.smoothBlock = myBlur()
        
        self.trainerG = torch.optim.Adam(self.transBlock.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.trainerD = torch.optim.Adam(self.D_B.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # FOR TESTING
    def forward(self, t, l):
        l = torch.tensor(l).float()
        tl = self.smoothBlock(t, l, self.gpu)
        label = l.repeat(1, 1, 1, 1)
        label = label.cuda() if self.gpu else label
        return self.transBlock(tl, label)
    
    # FOR TRAINING
    # init weight
    def init_networks(self, weights_init):
        self.transBlock.apply(weights_init)
        self.D_B.apply(weights_init)
        
    # WGAN-GP: calculate gradient penalty 
    def calc_gradient_penalty(self, netD, real_data, fake_data):
        alpha = torch.rand(real_data.shape[0], 1, 1, 1)
        alpha = alpha.cuda() if self.gpu else alpha

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = Variable(interpolates, requires_grad=True)

        disc_interpolates = netD(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() 
                              if self.gpu else torch.ones(disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    
    def update_discriminator(self, t, l):
        label = torch.tensor(l).float()
        label = label.repeat(t.size(0), 1, 1, 1)
        label = to_var(label) if self.gpu else label
        real_label = label.expand(label.size(0), label.size(1), t.size(2), t.size(3))    
        with torch.no_grad():
            tl = self.smoothBlock(t, l, self.gpu)
            fake_text = self.transBlock(tl, label)
            # print(tl.size(), real_label.size(), fake_text.size())
            fake_concat = torch.cat((tl, real_label, fake_text), dim=1)
        fake_output = self.D_B(fake_concat)
        real_concat = torch.cat((tl, real_label, t), dim=1)
        real_output = self.D_B(real_concat)
        gp = self.calc_gradient_penalty(self.D_B, real_concat.data, fake_concat.data)
        LBadv = self.lambda_adv*(fake_output.mean() - real_output.mean() + self.lambda_gp * gp)
        self.trainerD.zero_grad()
        LBadv.backward()
        self.trainerD.step()
        return (real_output.mean() - fake_output.mean()).data.mean() * self.lambda_adv
    
    def update_generator(self, t, l):
        label = torch.tensor(l).float()
        label = label.repeat(t.size(0), 1, 1, 1)
        label = label.cuda() if self.gpu else label
        real_label = label.expand(label.size(0), label.size(1), t.size(2), t.size(3)) 
        tl = self.smoothBlock(t, l, self.gpu)
        fake_text = self.transBlock(tl, label)
        fake_concat = torch.cat((tl, real_label, fake_text), dim=1)
        fake_output = self.D_B(fake_concat)
        LBadv = -fake_output.mean() * self.lambda_adv
        LBrec = self.loss(fake_text, t) * self.lambda_l1
        LB = LBadv + LBrec
        self.trainerG.zero_grad()
        LB.backward()
        self.trainerG.step()
        #global id
        #if id % 50 == 0:
        #    viz_img = to_data(torch.cat((t[0], tl[0], fake_text[0]), dim=2))
        #    save_image(viz_img, '../output/deblur_result%d.jpg'%id)
        #id += 1
        return LBadv.data.mean(), LBrec.data.mean()
    
    def one_pass(self, t, scales):
        l = random.choice(scales)
        LDadv = self.update_discriminator(t, l)
        LGadv, Lrec = self.update_generator(t, l)
        return [LDadv,LGadv,Lrec]
    
    
######################## ShapeMatchingGAN
class ShapeMatchingGAN(nn.Module):
    def __init__(self, GS_nlayers = 6, DS_nlayers = 5, GS_nf = 32, DS_nf = 32,
                 GT_nlayers = 6, DT_nlayers = 5, GT_nf = 32, DT_nf = 32, 
                 DG_nf = 32, DG_nlayers = 3, GG_nf = 32, GG_nlayers = 3, gpu=True):
        super(ShapeMatchingGAN, self).__init__()
        
        self.GS_nlayers = GS_nlayers
        self.DS_nlayers = DS_nlayers
        self.GS_nf = GS_nf
        self.DS_nf = DS_nf
        self.GT_nlayers = GT_nlayers
        self.DT_nlayers = DT_nlayers
        self.GT_nf = GT_nf
        self.DT_nf = DT_nf        
        self.gpu = gpu
        self.lambda_l1 = 100
        self.lambda_gp = 10
        self.lambda_sadv = 0.1
        self.lambda_gly = 1.0
        self.lambda_tadv = 1.0
        self.lambda_sty = 0.01
        self.style_weights = [1e3/n**2 for n in [64,128,256,512,512]]
        self.loss = nn.L1Loss()
        self.gramloss = GramMSELoss()
        self.gramloss = self.gramloss.cuda() if self.gpu else self.gramloss
        self.getmask = SemanticFeature()

        self.DG_nf = DG_nf
        self.DG_nlayers = DG_nlayers
        self.GG_nf = GG_nf
        self.GG_nlayers = GG_nlayers
        for param in self.getmask.parameters():
            param.requires_grad = False

        self.G_S = GlyphGenerator(self.GS_nf, self.GS_nlayers)
        self.D_S = Discriminator(3, self.DS_nf, self.DS_nlayers)
        self.G_T = TextureGenerator(self.GT_nf, self.GT_nlayers)
        self.D_T = Discriminator(6, self.DT_nf, self.DT_nlayers)
        self.D_G = Discriminator_gtf(self.DG_nf, self.DG_nlayers)
        self.G_G = Discriminator_gtf(self.GG_nf, self.GG_nlayers)
        
        self.trainerG_S = torch.optim.Adam(self.G_S.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.trainerD_S = torch.optim.Adam(self.D_S.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.trainerG_T = torch.optim.Adam(self.G_T.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.trainerD_T = torch.optim.Adam(self.D_T.parameters(), lr=0.0002, betas=(0.5, 0.999)) 
        self.trainerD_G = torch.optim.Adam(self.D_G.parameters(), lr=0.0002, betas=(0.5, 0.999))    
        self.trainerG_G = torch.optim.Adam(self.G_G.parameters(), lr=0.0002, betas=(0.5, 0.999))   
    
    # FOR TESTING
    def forward(self, x, l):
        x[:,0:1] = gaussian(x[:,0:1], stddev=0.2)
        xl = self.G_S(x, l) 
        xl[:,0:1] = gaussian(xl[:,0:1], stddev=0.2)
        return self.G_T(xl)
            
    # FOR TRAINING
    # init weight
    def init_networks(self, weights_init):
        #self.G_S.apply(weights_init)
        self.D_S.apply(weights_init)
        #self.G_T.apply(weights_init)
        self.D_T.apply(weights_init)
        self.D_G.apply(weights_init)
        self.G_G.apply(weights_init)
        
    # WGAN-GP: calculate gradient penalty 
    def calc_gradient_penalty(self, netD, real_data, fake_data):
        alpha = torch.rand(real_data.shape[0], 1, 1, 1)
        alpha = alpha.cuda() if self.gpu else alpha

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = Variable(interpolates, requires_grad=True)

        disc_interpolates = netD(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() 
                              if self.gpu else torch.ones(disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    
    def update_structure_discriminator(self, x, xl, l, labels,style_codem):   
        with torch.no_grad():
            fake_x = self.G_S(xl, l,labels,style_codem)
        fake_output = self.D_S(fake_x)
        real_output = self.D_S(x)
        gp = self.calc_gradient_penalty(self.D_S, x.data, fake_x.data)
        LSadv = self.lambda_sadv*(fake_output.mean() - real_output.mean() + self.lambda_gp * gp)
        self.trainerD_S.zero_grad()
        LSadv.backward()
        self.trainerD_S.step()
        return (real_output.mean() - fake_output.mean()).data.mean()*self.lambda_sadv

    def calc_gradient_penalty1(self, netD, real_data, fake_data):
        alpha = torch.rand(real_data.shape[0], 1, 1, 1)
        alpha = alpha.cuda() if self.gpu else alpha

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = Variable(interpolates, requires_grad=True)

        disc_interpolates = netD(interpolates,interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() 
                              if self.gpu else torch.ones(disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

# Discriminator for crop image
    def update_structure_gtf_discriminator(self, x, xl, l, labels,style_codem):
        w = random.randint(0,127)
        h = random.randint(0,127) 
        x_crop = x[:,:,w:w+128,h:h+128]
        #print(x_crop.size())  
        with torch.no_grad():
            fake_x = self.G_S(xl, l,labels,style_codem)
            fake_x_crop = fake_x[:,:,w:w+128,h:h+128]
        #print(fake_x_crop.size())
        fake_output = self.D_G(fake_x_crop,x_crop)
        real_output = self.D_G(x_crop,x_crop)
        gp = self.calc_gradient_penalty1(self.D_G, x_crop.data, fake_x_crop.data)
        LSgtf = self.lambda_sadv*(fake_output.mean() - real_output.mean() + self.lambda_gp * gp)
        self.trainerD_G.zero_grad()
        LSgtf.backward()
        self.trainerD_G.step()
        return (real_output.mean() - fake_output.mean()).data.mean()*self.lambda_sadv
    
    def update_structure_generator(self, x, xl, l,labels,style_codem, t=None):
        w = random.randint(0,127)
        h = random.randint(0,127) 
        x_crop = x[:,:,w:w+128,h:h+128]
        fake_x = self.G_S(xl, l,labels,style_codem)
        fake_x_crop = fake_x[:,:,w:w+128,h:h+128]
        fake_output = self.D_S(fake_x)
        fake_output1 = self.D_G(fake_x_crop,x_crop)
        LSadv1 = -fake_output1.mean()*self.lambda_sadv
        LSadv = -fake_output.mean()*self.lambda_sadv
        LSrec = self.loss(fake_x, x) * self.lambda_l1
        LS = LSadv + LSrec +LSadv1
        if t is not None:
            # weight map based on the distance field 
            # whose pixel value increases with its distance to the nearest text contour point of t
            Mt = (t[:,1:2]+t[:,2:3])*0.5+1.0
            t_noise = t.clone()
            t_noise[:,0:1] = gaussian(t_noise[:,0:1], stddev=0.2)
            fake_t = self.G_S(t_noise, l)
            LSgly = self.loss(fake_t*Mt, t*Mt) * self.lambda_gly
            LS = LS + LSgly
        self.trainerG_S.zero_grad()
        LS.backward()
        self.trainerG_S.step()
        #global id
        #if id % 60 == 0:
        #    viz_img = to_data(torch.cat((x[0], xl[0], fake_x[0]), dim=2))
        #    save_image(viz_img, '../output/structure_result%d.jpg'%id)
        #id += 1
        return LSadv.data.mean(), LSrec.data.mean(), LSadv1.data.mean(), LSgly.data.mean() if t is not None else 0
    
    def structure_one_pass(self, x, xl, l, labels,style_codem, t=None,):
        LDadv = self.update_structure_discriminator(x, xl, l, labels,style_codem)
        #LDgtf = self.update_structure_gtf_discriminator(x, xl, l, labels,style_codem)
        LGadv, Lrec, LSadv1, Lgly = self.update_structure_generator(x, xl, l, labels,style_codem, t)
        return [LDadv, LGadv, Lrec, Lgly, LSadv1]    
    
    def update_texture_discriminator(self, x, y, labels, style_codem):
        with torch.no_grad():
            fake_y = self.G_T(x,labels,style_codem)          
            fake_concat = torch.cat((x, fake_y), dim=1)
        fake_output = self.D_T(fake_concat)
        real_concat = torch.cat((x, y), dim=1)
        real_output = self.D_T(real_concat)
        gp = self.calc_gradient_penalty(self.D_T, real_concat.data, fake_concat.data)
        LTadv = self.lambda_tadv*(fake_output.mean() - real_output.mean() + self.lambda_gp * gp)
        self.trainerD_T.zero_grad()
        LTadv.backward()
        self.trainerD_T.step()
        return (real_output.mean() - fake_output.mean()).data.mean()*self.lambda_tadv        

    def update_texture_gtf_discriminator(self, x, y, labels,style_codem):
        w = random.randint(0,127)
        h = random.randint(0,127) 
        y_crop = y[:,:,w:w+128,h:h+128]
        with torch.no_grad():
            fake_y = self.G_T(x,labels,style_codem)          
            fake_y_crop = fake_y[:,:,w:w+128,h:h+128]
        fake_output = self.G_G(fake_y_crop,y_crop)
        real_output = self.G_G(y_crop,y_crop)
        gp = self.calc_gradient_penalty1(self.G_G, y_crop.data, fake_y_crop.data)
        LTadv = self.lambda_tadv*(fake_output.mean() - real_output.mean() + self.lambda_gp * gp)
        self.trainerG_G.zero_grad()
        LTadv.backward()
        self.trainerG_G.step()
        return (real_output.mean() - fake_output.mean()).data.mean()*self.lambda_tadv


    def update_texture_generator(self, x, y, labels,style_codem, t=None, l=None, VGGfeatures=None, style_targets=None):
        w = random.randint(0,127)
        h = random.randint(0,127) 
        x_crop = x[:,:,w:w+128,h:h+128]
        fake_y = self.G_T(x,labels,style_codem)
        fake_y_crop = fake_y[:,:,w:w+128,h:h+128]
        fake_concat = torch.cat((x, fake_y), dim=1)
        fake_output = self.D_T(fake_concat)
        fake_output1 = self.G_G(fake_y_crop,x_crop)
        LTadv = -fake_output.mean()*self.lambda_tadv
        LTadv1 = -fake_output1.mean()*self.lambda_sadv
        Lrec = self.loss(fake_y, y) * self.lambda_l1
        LT = LTadv + Lrec + LTadv1
        if t is not None:
            with torch.no_grad():
                t[:,0:1] = gaussian(t[:,0:1], stddev=0.2)
                source_mask = self.G_S(t, l).detach()
                source = source_mask.clone()
                source[:,0:1] = gaussian(source[:,0:1], stddev=0.2)
                smaps_fore = [(A.detach()+1)*0.5 for A in self.getmask(source_mask[:,0:1])]
                smaps_back = [1-A for A in smaps_fore]
            fake_t = self.G_T(source)
            out = VGGfeatures(fake_t)
            style_losses1 = [self.style_weights[a] * self.gramloss(A*smaps_fore[a], style_targets[0][a]) for a,A in enumerate(out)]
            style_losses2 = [self.style_weights[a] * self.gramloss(A*smaps_back[a], style_targets[1][a]) for a,A in enumerate(out)]
            Lsty = (sum(style_losses1)+ sum(style_losses2)) * self.lambda_sty
            LT = LT + Lsty
        #global id
        #if id % 20 == 0:
        #    viz_img = to_data(torch.cat((x[0], y[0], fake_y[0]), dim=2))
        #    save_image(viz_img, '../output/texturee_result%d.jpg'%id)
        #id += 1             
        self.trainerG_T.zero_grad()
        LT.backward()
        self.trainerG_T.step()   
        return LTadv.data.mean(), Lrec.data.mean(), LTadv1.data.mean(), Lsty.data.mean() if t is not None else 0
    
    def texture_one_pass(self, x, y, labels,style_codem, t=None, l=None, VGGfeatures=None, style_targets=None):
        LDadv = self.update_texture_discriminator(x, y, labels,style_codem)
        #LDgtf = self.update_texture_gtf_discriminator(x, y, labels,style_codem)
        LGadv, Lrec,LDadv1, Lsty = self.update_texture_generator(x, y, labels, t, l,style_codem, VGGfeatures, style_targets)
        return [LDadv, LGadv, Lrec, Lsty, LDadv1]
    
    def save_structure_model(self, filepath, filename):     
        torch.save(self.G_S.state_dict(), os.path.join(filepath, filename+'-GS.ckpt'))
        torch.save(self.D_S.state_dict(), os.path.join(filepath, filename+'-DS.ckpt'))
        torch.save(self.D_G.state_dict(), os.path.join(filepath, filename+'-DG.ckpt'))
    def save_texture_model(self, filepath, filename):
        torch.save(self.G_T.state_dict(), os.path.join(filepath, filename+'-GT.ckpt'))
        torch.save(self.D_T.state_dict(), os.path.join(filepath, filename+'-DT.ckpt'))
        torch.save(self.G_G.state_dict(), os.path.join(filepath, filename+'-GG.ckpt'))


class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout):
        #parser = TrainShapeMatchingOptions()
        parser = TestOptions()
        opts = parser.parse()
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        our_norm_type = 'spadesyncbatch3x3'
        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        #if 'spectral' in opt.norm_G:
        #    self.conv_0 = spectral_norm(self.conv_0)
        #    self.conv_1 = spectral_norm(self.conv_1)
        #    if self.learned_shortcut:
        #        self.conv_s = spectral_norm(self.conv_s)
        # define normalization layers
        spade_config_str = opts.norm_G.replace('spectral', '')
        normtype_list = ['spadeinstance3x3', 'spadesyncbatch3x3', 'spadebatch3x3']
        our_norm_type = 'spadesyncbatch3x3'
        spade_config_str0 = spectral_norm(self.conv_0)
        spade_config_str1 = spectral_norm(self.conv_1)
        #self.norm_0 = SPADE(spade_config_str,fin,opts.label_nc)
        #self.norm_1 = SPADE(spade_config_str,fmiddle,opts.label_nc)
        #if self.learned_shortcut:
        #    self.norm_s = SPADE(spade_config_str, fin, opt.semantic_nc)
        self.ace_0 = ACE(our_norm_type, fin, 3,spade_params=[spade_config_str, fin, 3],)
        self.ace_1 = ACE(our_norm_type, fmiddle, 3,spade_params=[spade_config_str, fin, 3],)
    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg,style_codem):
        x_s = self.shortcut(x, seg,style_codem)

        dx = self.ace_0(x, seg,style_codem)

        dx = self.conv_0(self.actvn(dx))
        dx = self.ace_0(dx, seg,style_codem)

        dx = self.conv_1(self.actvn(dx))

        out = x_s + dx

        return out

    def shortcut(self, x, seg,style_codem):
        if self.learned_shortcut:
            x_s = self.ace_s(x, seg, style_codem, obj_dic)
            x_s = self.conv_s(x_s)
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

# Creates SPADE normalization layer based on the given configuration
# SPADE consists of two steps. First, it normalizes the activations using
# your favorite normalization method, such as Batch Norm or Instance Norm.
# Second, it applies scale and bias to the normalized output, conditioned on
# the segmentation map.
# The format of |config_text| is spade(norm)(ks), where
# (norm) specifies the type of parameter-free normalization.
#       (e.g. syncbatch, batch, instance)
# (ks) specifies the size of kernel in the SPADE module (e.g. 3x3)
# Example |config_text| will be spadesyncbatch3x3, or spadeinstance5x5.
# Also, the other arguments are
# |norm_nc|: the #channels of the normalized activations, hence the output dim of SPADE
# |label_nc|: the #channels of the input semantic map, hence the input dim of SPADE



class Zencoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=2, norm_layer=nn.InstanceNorm2d):
        super(Zencoder, self).__init__()
        self.output_nc = output_nc
        input_nc=3
        model = [nn.ReflectionPad2d(0), nn.Conv2d(input_nc, ngf, kernel_size=3, padding=0),
                 norm_layer(ngf), nn.LeakyReLU(0.2, False)]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), nn.LeakyReLU(0.2, False)]

        ### upsample
        for i in range(1):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), nn.LeakyReLU(0.2, False)]

        model += [nn.ReflectionPad2d(1), nn.Conv2d(256, output_nc, kernel_size=3, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)


    def forward(self, input, segmap):
        self.model = self.model.cuda()
        codes = self.model(input)
        #print('code',codes.size())
        segmap = F.interpolate(segmap, size=codes.size()[2:], mode='nearest')

        # print(segmap.shape)
        # print(codes.shape)


        b_size = codes.shape[0]
        # h_size = codes.shape[2]
        # w_size = codes.shape[3]
        f_size = codes.shape[1]

        s_size = segmap.shape[1]

        codes_vector = torch.zeros((b_size, s_size, f_size), dtype=codes.dtype, device=codes.device)


        for i in range(b_size):
            for j in range(s_size):
                component_mask_area = torch.sum(segmap.bool()[i, j])

                if component_mask_area > 0:
                    codes_component_feature = codes[i].masked_select(segmap.bool()[i, j]).reshape(f_size,  component_mask_area).mean(1)
                    codes_vector[i][j] = codes_component_feature

                    # codes_avg[i].masked_scatter_(segmap.bool()[i, j], codes_component_mu)

        return codes_vector
