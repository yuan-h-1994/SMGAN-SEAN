from __future__ import print_function
import torch
from models import SketchModule, ShapeMatchingGAN,Zencoder
from utils import load_image, to_data, to_var, visualize, save_image, gaussian, weights_init
from utils import load_train_batchfnames, prepare_text_batch, load_style_image_pair, cropping_training_batches, label_input
import random
from options import TrainShapeMatchingOptions
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    # parse options
    parser = TrainShapeMatchingOptions()
    opts = parser.parse()
    myZencoder = Zencoder(1, 512)
    # create model
    print('--- create model ---')
    netShapeM = ShapeMatchingGAN(opts.GS_nlayers, opts.DS_nlayers, opts.GS_nf, opts.DS_nf,
                     opts.GT_nlayers, opts.DT_nlayers, opts.GT_nf, opts.DT_nf, opts.DG_nlayers, opts.DG_nf,
                     opts.GG_nlayers, opts.GG_nf, opts.gpu)
    netSketch = SketchModule(opts.GB_nlayers, opts.DB_nlayers, opts.GB_nf, opts.DB_nf, opts.gpu)


    if opts.gpu:
        netShapeM.cuda()
        netSketch.cuda()
    netShapeM.init_networks(weights_init)
    netShapeM.train()

    netSketch.load_state_dict(torch.load(opts.load_GB_name))
    netSketch.eval()

    print('--- training ---')
    # load image pair
    scales = [l*2.0//(opts.scale_num-1)-1 for l in range(opts.scale_num)]
    for epoch in range(opts.step1_epochs):
        for i in range(opts.Straining_num//opts.batchsize):
            for j in range(opts.style_number):
                j = str(j)
                label_name = '../data/style/label/'+ j +'.png'
                style_name = '../data/style/style/'+ j +'.png'
                labels = label_input(label_name,opts.batchsize//opts.style_number, opts.gpu)
                labels = to_var(labels) if opts.gpu else labels
                Xl, X, _, Noise = load_style_image_pair(style_name, scales, netSketch, opts.gpu)
                Xl = [to_var(a) for a in Xl] if opts.gpu else Xl
                X = to_var(X) if opts.gpu else X
                Noise = to_var(Noise) if opts.gpu else Noise
                idx = opts.scale_num-1           
                xl, x = cropping_training_batches(Xl[idx], X, Noise, opts.batchsize//opts.style_number, 
                                      opts.Sanglejitter, opts.subimg_size, opts.subimg_size)
                style_code =  myZencoder(input=x, segmap=labels)
                style_code =style_code[: ,0,:]
                if j == str(0):
                    xm = x
                    xlm = xl
                    labelm = labels   
                    style_codem = style_code
                if j != str(0):                      
                    xm = torch.cat((xm,x), dim = 0)
                    xlm = torch.cat((xlm,xl), dim = 0)
                    labelm = torch.cat((labelm,labels), dim = 0) 
                    style_codem = torch.cat((style_codem,style_code), dim = 0)
                #print('label',labelm.size())    
            losses = netShapeM.structure_one_pass(xm, xlm, scales[idx],labelm,style_codem)
            print('Step1, Epoch [%02d/%02d][%03d/%03d]' %(epoch+1, opts.step1_epochs, i+1, 
                                                          opts.Straining_num/opts.batchsize), end=': ')
            print('LDadv: %+.3f, LGadv: %+.3f, Lrec: %+.3f, Lgly: %+.3f'%(losses[0], losses[1], losses[2], losses[3]))
    netShapeM.G_S.myCopy()
    for epoch in range(opts.step2_epochs):
        for i in range(opts.Straining_num//opts.batchsize):
            for j in range(opts.style_number):
                j = str(j)
                label_name = '../data/style/label/'+ j +'.png'
                style_name = '../data/style/style/'+ j +'.png'
                labels = label_input(label_name,opts.batchsize//opts.style_number, opts.gpu)
                labels = to_var(labels) if opts.gpu else labels
                Xl, X, _, Noise = load_style_image_pair(style_name, scales, netSketch, opts.gpu)
                Xl = [to_var(a) for a in Xl] if opts.gpu else Xl
                X = to_var(X) if opts.gpu else X
                Noise = to_var(Noise) if opts.gpu else Noise
                idx = random.choice([0, opts.scale_num-1])
                xl, x = cropping_training_batches(Xl[idx], X, Noise, opts.batchsize//opts.style_number, 
                                      opts.Sanglejitter, opts.subimg_size, opts.subimg_size)
                style_code =  myZencoder(input=x, segmap=labels)
                style_code =style_code[: ,0,:]
                if j == str(0):
                    xm = x
                    xlm = xl
                    labelm = labels
                    style_codem = style_code     
                if j != str(0):                      
                    xm = torch.cat((xm,x), dim = 0)
                    xlm = torch.cat((xlm,xl), dim = 0)
                    labelm = torch.cat((labelm,labels), dim = 0) 
                    style_codem = torch.cat((style_codem,style_code), dim = 0)                   
            losses = netShapeM.structure_one_pass(xm, xlm, scales[idx],labelm,style_codem)
            print('Step2, Epoch [%02d/%02d][%03d/%03d]' %(epoch+1, opts.step2_epochs, i+1, 
                                                          opts.Straining_num/opts.batchsize), end=': ')
            print('LDadv: %+.3f, LGadv: %+.3f, Lrec: %+.3f, Lgly: %+.3f'%(losses[0], losses[1], losses[2], losses[3]))
    for epoch in range(opts.step3_epochs):
        for i in range(opts.Straining_num//opts.batchsize):
            for j in range(opts.style_number):
                j = str(j)
                label_name = '../data/style/label/'+ j +'.png'
                style_name = '../data/style/style/'+ j +'.png'
                labels = label_input(label_name,opts.batchsize//opts.style_number, opts.gpu)
                labels = to_var(labels) if opts.gpu else labels
                Xl, X, _, Noise = load_style_image_pair(style_name, scales, netSketch, opts.gpu)
                Xl = [to_var(a) for a in Xl] if opts.gpu else Xl
                X = to_var(X) if opts.gpu else X
                Noise = to_var(Noise) if opts.gpu else Noise
                idx = random.choice(range(opts.scale_num))
                xl, x = cropping_training_batches(Xl[idx], X, Noise, opts.batchsize//opts.style_number, 
                                      opts.Sanglejitter, opts.subimg_size, opts.subimg_size)
                style_code =  myZencoder(input=x, segmap=labels)
                style_code =style_code[: ,0,:]
                if j == str(0):
                    xm = x
                    xlm = xl
                    labelm = labels 
                    style_codem = style_code    
                if j != str(0):                      
                    xm = torch.cat((xm,x), dim = 0)
                    xlm = torch.cat((xlm,xl), dim = 0)
                    labelm = torch.cat((labelm,labels), dim = 0)
                    style_codem = torch.cat((style_codem,style_code), dim = 0)
            losses = netShapeM.structure_one_pass(xm, xlm, scales[idx],labelm,style_codem)  
            print('Step3, Epoch [%02d/%02d][%03d/%03d]' %(epoch+1, opts.step3_epochs, i+1, 
                                                          opts.Straining_num/opts.batchsize), end=': ')
            print('LDadv: %+.3f, LGadv: %+.3f, Lrec: %+.3f, Lgly: %+.3f'%(losses[0], losses[1], losses[2], losses[3]))
    if opts.glyph_preserve:
        fnames = load_train_batchfnames(opts.text_path, opts.batchsize, 
                                        opts.text_datasize, opts.Straining_num)
        for epoch in range(opts.step4_epochs):
            itr = 0
            for fname in fnames:
                itr += 1
                t = prepare_text_batch(fname, anglejitter=False)
                idx = random.choice(range(opts.scale_num))
                xl, x = cropping_training_batches(Xl[idx], X, Noise, opts.batchsize, 
                                          opts.Sanglejitter, opts.subimg_size, opts.subimg_size)
                t = to_var(x) if opts.gpu else t
                losses = netShapeM.structure_one_pass(x, xl, scales[idx],labels, t)  
                print('Step4, Epoch [%02d/%02d][%03d/%03d]' %(epoch+1, opts.step4_epochs, itr+1, 
                                                          len(fnames)), end=': ')
                print('LDadv: %+.3f, LGadv: %+.3f, Lrec: %+.3f, Lgly: %+.3f'%(losses[0], losses[1], losses[2], losses[3])) 

    print('--- save ---')
    # directory
    netShapeM.save_structure_model(opts.save_path, opts.save_name)    

if __name__ == '__main__':
    main()