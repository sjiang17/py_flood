from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
# from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from flood_models import build_UNet
from read_featuremap_test import FeatureReader
# from my_loss import L1Loss

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

lr = 0.01
training_name = 'PEDINST_MASK_kitti_trans_lr{}'.format(lr)
# training_name = 'test3'

data_dir = '/pvdata/dataset/image_test_loss/no_resize/feature_map-conv4pool/image_unocc_untrunc'
featuremap_datasets = FeatureReader(data_dir)
dataloaders = torch.utils.data.DataLoader(featuremap_datasets, batch_size=1,
                                                shuffle=False, num_workers=0)
dataset_sizes = len(featuremap_datasets)
print (dataset_sizes)

use_gpu = torch.cuda.is_available()

save_dir = os.path.join('save', training_name)
if not os.path.exists(save_dir):
	os.makedirs(save_dir)


def train_model(model, criterion, optimizer, num_epochs=200):
    since = time.time()
    
    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        summary_file = open(os.path.join(save_dir, 'log.txt'), 'a')
        summary_file.write("Epoch {} ".format(epoch))
        
        # Each epoch has a training and validation phase
        for phase in ['test']:
            if phase == 'train':
                # scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for ix, data in enumerate(dataloaders):
                # get the inputs
                inputs, gt = data
                # wrap them in Variable
                if use_gpu:
                    if phase == 'train':
                        inputs = Variable(inputs.cuda())
                        gt = Variable(gt.cuda())
                    else:
                        inputs = Variable(inputs.cuda(), volatile=True)
                        gt = Variable(gt.cuda(), volatile=True)
                else:
                    if phase == 'train':
                        inputs, gt = Variable(inputs), Variable(gt)
                    else:
                        inputs, gt = Variable(inputs, volatile=True), Variable(gt, volatile=True)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                # print (type(outputs))
                # print (outputs.size())
                
                loss = criterion(outputs, gt)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                
                iter_loss = loss.data[0] * inputs.size(0)
                running_loss += iter_loss
                
                if ix % 100 == 0:
                    print ('iter {}, Loss = {:.4f}'.format(ix, iter_loss))

            epoch_loss = running_loss / dataset_sizes
            
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            
            summary_file.write("{} loss {:.4f} ".format(phase, epoch_loss))
            
            # if epoch == 1 or epoch % 5 == 0:
            #     save_name = 'transformer_{}_{}.pth'.format(training_name ,epoch)
            #     torch.save(model.state_dict(), os.path.join(save_dir, save_name))
        
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

pretrained_model = '/siyuvol/py_flood/save/MASK_kitti_UNet_lr0.01_SGD_dropout/transformer_MASK_kitti_UNet_lr0.01_SGD_dropout_250.pth'
model_trans = build_UNet(type='UNet1', use_dropout=True, pretrained_model=pretrained_model)
if use_gpu:
    model_trans = model_trans.cuda()

criterion = nn.L1Loss()
optimizer_trans = optim.SGD(model_trans.parameters(), lr=lr, momentum=0.9, weight_decay=0)
# optimizer_trans = optim.Adam(model_trans.parameters(), lr=lr, weight_decay=0.0005)

# Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

print(training_name)
train_model(model_trans, criterion, optimizer_trans, num_epochs=1)
