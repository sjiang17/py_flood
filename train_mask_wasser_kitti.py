from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
# from torch.optim import lr_scheduler
from torch.autograd import Variable
import cv2
import numpy as np
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from flood_models import build_UNet
from read_featuremap_occlusion import FeatureReader
from D_Net import build_DNet
from my_loss import L1Loss

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

lr = 1e-3
lr_d = 1e-6
lmda = 1
rratio = 3.0
training_name = 'test_wasser'
# training_name = 'MASKED_kitti_wasser_lrg{}_lrd{}_lmda{}_r{}_SHALLOW_DROP'.format(lr, lr_d, lmda, rratio)
BATCH_SIZE = 1

data_dir = '/siyuvol/dataset/kitti/mask_noresize/feature_map-conv4pool/'
featuremap_datasets = {x: FeatureReader(os.path.join(data_dir, x))
                                          for x in ['train', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(featuremap_datasets[x], batch_size=BATCH_SIZE,
                                                shuffle=False, num_workers=4)
                                                for x in ['train', 'test']}
dataset_sizes = {x: len(featuremap_datasets[x]) for x in ['train', 'test']}
print (dataset_sizes)

use_gpu = torch.cuda.is_available()

save_dir = os.path.join('save', training_name)
if not os.path.exists(save_dir):
	os.makedirs(save_dir)

def myGetVariable(a, use_gpu, phase):
    if use_gpu:
        if phase == 'train':
            return Variable(a.cuda())
        else:
            return Variable(a.cuda(), volatile=True)
    else:
        if phase == 'train':
            return Variable(a)
        else:
            return Variable(a, volatile=True)

def construct_mask(unocc_cords, output_size):
    bs, ch, h, w = output_size
    mask = torch.ones(h, w)
    for unocc_cord in unocc_cords:
        xmin = int(np.rint(w * unocc_cord[0]))
        ymin = int(np.rint(h * unocc_cord[1]))
        xmax = int(np.rint(w * unocc_cord[2]))
        ymax = int(np.rint(h * unocc_cord[3]))
        if xmax > xmin and ymax > ymin:
            mask[ymin:ymax, xmin:xmax] = 0
    mask = torch.stack([mask for mm in range(ch)], 0)
    mask = mask.unsqueeze(0)
    return mask


def train_model(netG, netD, criterion_rec, optimizer_trans, optimizer_D, num_epochs):
    since = time.time()

    label_adv_tensor = torch.FloatTensor(BATCH_SIZE)
    
    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        summary_file = open(os.path.join(save_dir, 'log.txt'), 'a')
        summary_file.write("Epoch {} ".format(epoch))
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                # scheduler.step()
                netG.train(True)  # Set model to training mode
                netD.train(True)
            else:
                netG.train(False)  # Set model to evaluate mode
                netD.train(False)

            running_loss_trans = 0.0
            running_loss_adv_real = 0.0
            running_loss_adv_fake = 0.0
            running_loss_gen = 0.0
            
            acc_d1, acc_d2 = 0.0, 0.0
            unocc_cnt = 0.0
            acc_d1_epoch, acc_d2_epoch = 0.0, 0.0
            unocc_cnt_epoch = 0.0

            one = torch.FloatTensor([1])
            mone = one * -1
            one, mone = one.cuda(), mone.cuda()
            
            # Iterate over data.
            for ix, data in enumerate(dataloaders[phase]):
                # get the inputs
                inputs, gt, occ_level, unocc_cords = data
                inputs = myGetVariable(inputs, use_gpu, phase)
                gt = myGetVariable(gt, use_gpu, phase)
                occ_level = occ_level.numpy()[0]
                # print (type(occ), occ)
                #################################
                # Update D network              #
                #################################

                #-------train with real-------
                # only when all objects are unoccluded
                for p in netD.parameters():
                    p.requires_grad = True 
                    p.data.clamp_(-0.01, 0.01)

                optimizer_D.zero_grad()

                if occ_level == 0:
                    unocc_cnt += 1
                    unocc_cnt_epoch += 1
                    # label_adv_real = myGetVariable(label_adv_tensor.fill_(1), use_gpu, phase)
                    outputs_D_real = netD(gt)
                    od1 = outputs_D_real.data.cpu()[0,0,0,0]
                    # print('od1: ', od1)
                    acc_d1 += (od1 >= 0.5)
                    acc_d1_epoch += (od1 >= 0.5)
                    
                    # loss_D_real = criterion_adv(outputs_D_real, label_adv_real) * rratio
                    if phase == 'train':
                        # loss_D_real.backward()
                        outputs_D_real.backward(one)
                    iter_loss_adv_real = outputs_D_real.data[0] * inputs.size(0)
                    
                else:
                    iter_loss_adv_real = 0.0
                
                #--------train with fake--------
                outputs_trans = netG(inputs) # fake
                # label_adv_fake = myGetVariable(label_adv_tensor.fill_(0), use_gpu, phase)
                outputs_D_fake = netD(outputs_trans.detach())
                od2 = outputs_D_fake.data.cpu()[0,0,0,0]
                acc_d2 += (od2 < 0.5)
                acc_d2_epoch += (od2 < 0.5)
                
                # loss_D_fake = criterion_adv(outputs_D_fake, label_adv_fake)
                if phase == 'train':
                    # loss_D_fake.backward()
                    outputs_D_fake.backward(mone)
                    errD = outputs_D_real - outputs_D_fake
                    optimizer_D.step()
                
                iter_loss_adv_fake = outputs_D_fake.data[0] * inputs.size(0)
                
                #################################
                # Update transformation network #
                #################################

                for p in netD.parameters():
                    p.requires_grad = False 
                # zero the parameter gradients
                optimizer_trans.zero_grad()

                # forward
                label_adv = myGetVariable(label_adv_tensor.fill_(0), use_gpu, phase) # we want to generator to produce 1
                outputs_D = netD(outputs_trans)
                
                if not occ_level == 2:
                    mask = construct_mask(unocc_cords, outputs_trans.size())
                    mask = torch.autograd.Variable(mask.cuda(), requires_grad=False)
                    loss_gen = criterion_rec(outputs_trans, gt, mask)
                    loss_trans = loss_gen + lmda * (outputs_D - label_adv)
                else:
                    # loss_trans = lmda * criterion_adv(outputs_D, label_adv)
                    lmda * (outputs_D - label_adv)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss_trans.backward()
                    optimizer_trans.step()
                
                iter_loss_trans = loss_trans.data[0] * inputs.size(0)
                
                iter_loss_gen = loss_gen.data[0] * inputs.size(0)
                
                if ix % 100 == 0:
                    print ('iter {}, Loss Trans={:.4f}, Gen={:.4f}, Adv Real={:.8f}, Adv Fake={:.8f}, acc_d1: {:.2f}, acc_d2: {:.2f}'.format(
                        ix, iter_loss_trans, iter_loss_gen, iter_loss_adv_real, iter_loss_adv_fake, acc_d1/unocc_cnt, acc_d2/100.0))
                    # print ('unocc_cnt: {}'.format(unocc_cnt))
                    acc_d1, acc_d2 = 0.0, 0.0
                    unocc_cnt = 0.0
                    
                # statistics
                running_loss_trans += iter_loss_trans
                running_loss_adv_real += iter_loss_adv_real
                running_loss_adv_fake += iter_loss_adv_fake
                running_loss_gen += iter_loss_gen

            epoch_loss_trans = running_loss_trans / dataset_sizes[phase]
            epoch_loss_adv_real = running_loss_adv_real / dataset_sizes[phase]
            epoch_loss_adv_fake = running_loss_adv_fake / dataset_sizes[phase]
            epoch_loss_gen = running_loss_gen / dataset_sizes[phase]
            acc_d1_epoch /= unocc_cnt_epoch
            acc_d2_epoch /= dataset_sizes[phase]
            
            print('{} Loss Trans={:.4f}, Gen={:.4f}, Adv Real={:.8f}, Adv Fake={:.8f}, acc_d1={:.4f}, acc_d2={:.4f}'.format(
                phase, epoch_loss_trans, epoch_loss_gen, epoch_loss_adv_real, epoch_loss_adv_fake, acc_d1_epoch, acc_d2_epoch))
            
            summary_file.write("{} Loss Trans={:.4f}, Gen={:.4f}, Adv Real={:.8f}, Adv Fake={:.8f}, acc_d1={:.4f}, acc_d2={:.4f} ".format(
                phase, epoch_loss_trans, epoch_loss_gen, epoch_loss_adv_real, epoch_loss_adv_fake, acc_d1_epoch, acc_d2_epoch))
            
            if epoch == 1 or epoch % 5 == 0:
                save_name = 'transformer_{}_{}.pth'.format(training_name, epoch)
                torch.save(netG.state_dict(), os.path.join(save_dir, save_name))
                save_name_netD = 'netD_{}_{}.pth'.format(training_name, epoch)
                torch.save(netD.state_dict(), os.path.join(save_dir, save_name_netD))
        
        summary_file.write("\n")
        summary_file.close()
        print()
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # print('Best test Acc: {:4f}'.format(best_acc))

    # load best model weights
    # model.load_state_dict(best_model_wts)
    return

model_trans = build_UNet(type='UNet1',use_bias=True, use_dropout=True, is_pretrained=True)
model_D = build_DNet(is_shallow=True, use_dropout=True, is_wasser=True)
if use_gpu:
    model_trans = model_trans.cuda()
    model_D = model_D.cuda()

criterion_rec = L1Loss()
# criterion_adv = nn.BCELoss()
# optimizer_trans = optim.SGD(model_trans.parameters(), lr=lr, momentum=0.9)
# optimizer_trans = optim.Adam(model_trans.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_trans = optim.RMSprop(model_trans.parameters(), lr=lr)
optimizer_D = optim.SGD(model_D.parameters(), lr=lr_d, momentum=0.9)
#optimizer_D = optim.Adam(model_D.parameters(), lr=lr_d, betas=(0.5, 0.999), weight_decay=0.0005)

# Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
print(training_name)
train_model(model_trans, model_D, criterion_rec, optimizer_trans, optimizer_D, num_epochs=200)
