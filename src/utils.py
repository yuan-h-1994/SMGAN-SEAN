import torch

from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import cv2
import scipy.ndimage as pyimg
import random
import os
import torch.nn as nn
import torch.nn.functional as F
import re
pil2tensor = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5,0.5,0.5])])
tensor2pil = transforms.ToPILImage()

def to_var(x):
    """Convert tensor to variable."""
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def to_data(x):
    """Convert variable to tensor."""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data

def visualize(img_arr):
    plt.imshow(((img_arr.numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
    plt.axis('off')
    
def load_image(filename, load_type=0):
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5,0.5,0.5]),
    ])
    
    if load_type == 0:
        img = Image.open(filename)
    else:
        img = text_image_preprocessing(filename)
    img = transform(img)
        
    return img.unsqueeze(dim=0)

def save_image(img, filename):
    tmp = ((img.numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)
    cv2.imwrite(filename, cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR))

# black and white text image to distance-based text image
def text_image_preprocessing(filename):
    I = np.array(Image.open(filename))
    BW = I[:,:,0] > 127
    G_channel = pyimg.distance_transform_edt(BW)
    G_channel[G_channel>32]=32
    B_channel = pyimg.distance_transform_edt(1-BW)
    B_channel[B_channel>200]=200
    I[:,:,1] = G_channel.astype('uint8')
    I[:,:,2] = B_channel.astype('uint8')
    return Image.fromarray(I)

def gaussian(ins, mean = 0, stddev = 0.2):
    noise = ins.data.new(ins.size()).normal_(mean, stddev)
    return torch.clamp(ins + noise, -1, 1)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('my') == -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
# prepare batched filenames of all training data
# [[list of file names in one batch],[list of file names in one batch],...,[]]
def load_train_batchfnames(path, batch_size, usenum=3, trainnum=100000):
    trainnum = int((int(trainnum) // int(batch_size)) * batch_size)
    fnames = [('%04d.png' % (i%usenum)) for i in range(trainnum)]
    random.shuffle(fnames)
    trainbatches = [([]) for _ in range(trainnum//batch_size)]
    count = 0
    for i in range(trainnum//batch_size):
        traindatas = []
        for j in range(batch_size):
            traindatas += [os.path.join(path, fnames[count])]
            count = count + 1
        trainbatches[i] += traindatas
    return trainbatches

# prepare a batch of text images {t}
def prepare_text_batch(batchfnames, wd=256, ht=256, anglejitter=False):
    img_list = []
    for fname in batchfnames:
        img = Image.open(fname)     
        ori_wd, ori_ht = img.size
        w = random.randint(0,ori_wd-wd)
        h = random.randint(0,ori_ht-ht)
        img = img.crop((w,h,w+wd,h+ht))
        if anglejitter:
            random_angle = 90 * random.randint(0,3)
            img = img.rotate(random_angle)
        img = pil2tensor(img)         
        img = img.unsqueeze(dim=0)
        img_list.append(img)
    return torch.cat(img_list, dim=0)

# prepare {Xl, X, Y, fixed_noise} in PIL format from one image pair [X,Y]
def load_style_image_pair(filename, scales=[-1.0,-1.//3,1.//3,1.0], sketchmodule=None, gpu=True):
    img = Image.open(filename) 
    ori_wd, ori_ht = img.size
    ori_wd = ori_wd // 2
    X = pil2tensor(img.crop((0,0,ori_wd,ori_ht))).unsqueeze(dim=0)
    Y = pil2tensor(img.crop((ori_wd,0,ori_wd*2,ori_ht))).unsqueeze(dim=0)
    Xls = []    
    Noise = torch.tensor(0).float().repeat(1, 1, 1).expand(3, ori_ht, ori_wd)
    Noise = Noise.data.new(Noise.size()).normal_(0, 0.2)
    Noise = Noise.unsqueeze(dim=0)
    #Noise = tensor2pil((Noise+1)/2)    
    if sketchmodule is not None:
        X_ = to_var(X) if gpu else X
        for l in scales:  
            with torch.no_grad():
                Xl = sketchmodule(X_, l).detach()
            Xls.append(to_data(Xl) if gpu else Xl)            
    return [Xls, X, Y, Noise]

# create one-hot label map
def label_input(filename, batchsize, gpu=True):
    FloatTensor = torch.cuda.FloatTensor if gpu else torch.FloatTensor
    loader = transforms.Compose([transforms.ToTensor()])
    #print(filename)
    #label_map = Image.open(filename).resize((256,256))
    label_map = Image.open(filename).resize((320,320))
    w, h = label_map.size
    #if batchsize==2 or batchsize==6:
    if batchsize==16 or batchsize==4:
        bs = batchsize
    else:
        bs = 1
    nc = 3
    #nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
    #    else self.opt.label_nc
    labels = loader(label_map).unsqueeze(0)
    labels = labels.long()
    if gpu:
        labels = labels.cuda()
    input_label = FloatTensor(1, nc, h, w).zero_()
    input_semantics = input_label.scatter_(1, labels, 1.0)
    input_semantics = input_semantics.expand(bs,nc,h,w)
    return input_semantics

def rotate_tensor(x, angle):
    if angle == 1:
        return x.transpose(2, 3).flip(2)
    elif angle == 2:
        return x.flip(2).flip(3)
    elif angle == 3:
        return x.transpose(2, 3).flip(3)
    else:
        return x
    
# crop subimages for training 
# for structure transfer:  [Input,Output]=[Xl, X]
# for texture transfer:  [Input,Output]=[X, Y]
def cropping_training_batches(Input, Output, Noise, batchsize, anglejitter=False, wd=256, ht=256):
    img_list = []
    ori_wd = Input.size(2)
    ori_ht = Input.size(3)
    for i in range(batchsize):
        w = random.randint(0,ori_wd-wd)
        h = random.randint(0,ori_ht-ht)
        input = Input[:,:,w:w+wd,h:h+ht].clone()
        output = Output[:,:,w:w+wd,h:h+ht]
        noise = Noise[:,:,w:w+wd,h:h+ht]
        if anglejitter:
            random_angle = random.randint(0,3)
            input = rotate_tensor(input, random_angle)
            output = rotate_tensor(output, random_angle)
            noise = rotate_tensor(noise, random_angle)        
        input[:,0] = torch.clamp(input[:,0] + noise[:,0], -1, 1)        
        img_list.append(torch.cat((input,output), dim = 1))        
    data = torch.cat(img_list, dim=0)
    ins = data[:,0:3,:,:]
    outs = data[:,3:,:,:]
    return ins, outs






class ACE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc, ACE_Name=None, status='train', spade_params=None, use_rgb=True):
        super().__init__()

        self.ACE_Name = ACE_Name
        self.status = status
        self.save_npy = True
        self.Spade = SPADE(*spade_params)
        self.use_rgb = use_rgb
        self.style_length = 512
        self.blending_gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.blending_beta = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.noise_var = nn.Parameter(torch.zeros(norm_nc), requires_grad=True)

        #assert config_text.startswith('spade')
        #parsed = re.search('spade(\D+)(\d)x\d', config_text)
        #param_free_norm_type = str(parsed.group(1))
        #ks = int(parsed.group(2))
        #pw = ks // 2


        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)

        # The dimension of the intermediate embedding space. Yes, hardcoded.


        self.create_gamma_beta_fc_layers()

        self.conv_gamma = nn.Conv2d(self.style_length, norm_nc, kernel_size=5, padding=2)
        self.conv_beta = nn.Conv2d(self.style_length, norm_nc, kernel_size=5, padding=2)




    def forward(self, x, segmap, style_codem, obj_dic=None):

        # Part 1. generate parameter-free normalized activations
        added_noise = (torch.randn(x.shape[0], x.shape[3], x.shape[2], 1).cuda() * self.noise_var).transpose(1, 3)
        normalized = self.param_free_norm(x + added_noise)
        
        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        #print(style_codem.size())

        [b_size, f_size, h_size, w_size] = normalized.shape
        middle_avg = torch.zeros((b_size, self.style_length, h_size, w_size), device=normalized.device)

        gamma_avg = self.conv_gamma(middle_avg)
        beta_avg = self.conv_beta(middle_avg)
        #print(gamma_avg.size())
        gamma_spade, beta_spade = self.Spade(segmap)

        gamma_alpha = F.sigmoid(self.blending_gamma)
        beta_alpha = F.sigmoid(self.blending_beta)

        gamma_final = gamma_alpha * gamma_avg + (1 - gamma_alpha) * gamma_spade
        beta_final = beta_alpha * beta_avg + (1 - beta_alpha) * beta_spade
        out = normalized * (1 + gamma_final) + beta_final

        return out
    
    def create_gamma_beta_fc_layers(self):


        ###################  These codes should be replaced with torch.nn.ModuleList

        style_length = self.style_length

        self.fc_mu0 = nn.Linear(style_length, style_length)
        self.fc_mu1 = nn.Linear(style_length, style_length)
        self.fc_mu2 = nn.Linear(style_length, style_length)
        self.fc_mu3 = nn.Linear(style_length, style_length)
        self.fc_mu4 = nn.Linear(style_length, style_length)
        self.fc_mu5 = nn.Linear(style_length, style_length)
        self.fc_mu6 = nn.Linear(style_length, style_length)
        self.fc_mu7 = nn.Linear(style_length, style_length)
        self.fc_mu8 = nn.Linear(style_length, style_length)
        self.fc_mu9 = nn.Linear(style_length, style_length)
        self.fc_mu10 = nn.Linear(style_length, style_length)
        self.fc_mu11 = nn.Linear(style_length, style_length)
        self.fc_mu12 = nn.Linear(style_length, style_length)
        self.fc_mu13 = nn.Linear(style_length, style_length)
        self.fc_mu14 = nn.Linear(style_length, style_length)
        self.fc_mu15 = nn.Linear(style_length, style_length)
        self.fc_mu16 = nn.Linear(style_length, style_length)
        self.fc_mu17 = nn.Linear(style_length, style_length)
        self.fc_mu18 = nn.Linear(style_length, style_length)

class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc):
        super().__init__()

        config_text = 'spadebatch3x3'
        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )

        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, segmap):

        inputmap = segmap

        actv = self.mlp_shared(inputmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        return gamma, beta

