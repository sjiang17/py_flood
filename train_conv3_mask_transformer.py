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
from read_featuremap_occlusion import FeatureReader
from my_loss import L1Loss

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

lr = 0.01
# training_name = 'Conv3_GREYMASK_kitti_trans_lr{}'.format(lr)
training_name = 'test_conv3_worker4'

data_dir = '/siyuvol/dataset/kitti/greymask/feature_map-conv3pool/'
featuremap_datasets = {x: FeatureReader(os.path.join(data_dir, x))
                                          for x in ['train', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(featuremap_datasets[x], batch_size=1,
                                                shuffle=False, num_workers=4)
                                                for x in ['train', 'test']}
dataset_sizes = {x: len(featuremap_datasets[x]) for x in ['train', 'test']}
print (dataset_sizes)

use_gpu = torch.cuda.is_available()

save_dir = os.path.join('save', training_name)
if not os.path.exists(save_dir):
	os.makedirs(save_dir)


def train_model(model, criterion, optimizer, num_epochs=200):
    since = time.time()
    
    for epoch in range(96, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        summary_file = open(os.path.join(save_dir, 'log.txt'), 'a')
        summary_file.write("Epoch {} ".format(epoch))
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                # scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for ix, data in enumerate(dataloaders[phase]):
                # get the inputs
                inputs, gt, occ_level, occ_cords = data
                
                if occ_level.numpy()[0] == 2:
                    continue

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
                
                bs, ch, h, w = outputs.size()
                # occ_cords = [(0.03, 0.24, 0.63, 0.45), (0.32, 0.84, 0.43, 0.95)]
                mask = torch.ones(h, w)
                for occ_cord in occ_cords:
                    xmin = int(np.rint(w * occ_cord[0]))
                    ymin = int(np.rint(h * occ_cord[1]))
                    xmax = int(np.rint(w * occ_cord[2]))
                    ymax = int(np.rint(h * occ_cord[3]))
                    if xmax > xmin and ymax > ymin:
                        # print (xmax-xmin, ymax-ymin)
                        mask[ymin:ymax, xmin:xmax] = 0
                    # else:
                        # print (xmax-xmin, ymax-ymin)
                mask = torch.stack([mask for mm in range(ch)], 0)
                mask = mask.unsqueeze(0)
                mask = torch.autograd.Variable(mask.cuda(), requires_grad=False)
                
                loss = criterion(outputs, gt, mask)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                
                iter_loss = loss.data[0] * inputs.size(0)
                running_loss += iter_loss
                
                if ix % 100 == 0:
                    print ('iter {}, Loss = {:.4f}'.format(ix, iter_loss))

            epoch_loss = running_loss / dataset_sizes[phase]
            
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            
            summary_file.write("{} loss {:.4f} ".format(phase, epoch_loss))
            
            if epoch == 1 or epoch % 5 == 0:
                save_name = 'transformer_{}_{}.pth'.format(training_name ,epoch)
                torch.save(model.state_dict(), os.path.join(save_dir, save_name))
        
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

model_trans = build_UNet(type='UNet1_Conv3', use_dropout=True, is_pretrained=True)
if use_gpu:
    model_trans = model_trans.cuda()

criterion = L1Loss()
optimizer_trans = optim.SGD(model_trans.parameters(), lr=lr, momentum=0.9, weight_decay=0)
# optimizer_trans = optim.Adam(model_trans.parameters(), lr=lr, weight_decay=0.0005)

# Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

print(training_name)
train_model(model_trans, criterion, optimizer_trans, num_epochs=450)
