from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets
import time
import os
import copy
import datetime
from flood_models import build_UNet
from read_featuremap_test import FeatureReader

def nn_search(model, criterion, optimizer, num_epochs=1):
    since = time.time()
    neighbor_vec = np.zeros(dataset_sizes, dataset_sizes)
    model.train(False)

    for ixb, base_data in enumerate(base_dataloaders):
    	print ("base_data {} {}".format(ixb, ind_b))
    	summary_file = open(os.path.join(save_dir, 'log.txt'), 'a')
        summary_file.write("{} base_data {} {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), ixb, ind_b))

    	base_fm, base_name = base_data
        outputs = model(base_fm)
        
        # Iterate over data.
        for ix, data in enumerate(target_dataloaders):
            # get the inputs
            target_fm, target_name = data
            # wrap them in Variable
            if use_gpu:
                base_fm, target_fm = Variable(base_fm.cuda(), volatile=True),\
                						Variable(target_fm.cuda(), volatile=True)
            else:
                base_fm, target_fm = Variable(base_fm, volatile=True), \
                						Variable(target_fm, volatile=True)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            loss = criterion(outputs, target_fm)

            iter_loss = loss.data[0] #* base_fm.size(0)
            neighbor_vec[ixb, ix] = iter_loss
            if ix % 100 == 0:
                print ('{} iter {}, Loss = {:.4f}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), ix, iter_loss))

        nearests = np.argsort(neighbor_vec[ixb])[:5]
        for n in list(nearests):
        	print ("{} {}",format(n, neighbor_vec[ixb, n]))
        	summary_file.write("{} {}",format(n, neighbor_vec[ixb, n]))
        summary_file.write("\n\n\n")
        summary_file.close()
        print()
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
use_gpu = torch.cuda.is_available()

base_data_dir = '/siyuvol/dataset/kitti/mask_noresize/feature_map-conv4pool/nnsearch_example/0'
target_data_dir = '/siyuvol/dataset/kitti/mask_noresize/feature_map-conv4pool/nnsearch_example/1'

target_datasets = FeatureReader(target_data_dir)
dataset_sizes = len(featuremap_datasets)
target_dataloaders = torch.utils.data.DataLoader(target_datasets, batch_size=1,
                                                shuffle=False, num_workers=4)

base_datasets = FeatureReader(base_data_dir)
assert dataset_sizes == len(base_datasets)
base_dataloaders = torch.utils.data.DataLoader(base_datasets, batch_size=1,
                                                shuffle=False, num_workers=4)

os.path.join('save', 'nnsearch_MASK_kitti_UNet_lr0.01_SGD_dropout_250')
if not os.path.exists(save_dir):
	os.makedirs(save_dir)
pretrained_model = '/pvdata/savemodel/MASK_kitti_UNet_lr0.01_SGD_dropout/transformer_MASK_kitti_UNet_lr0.01_SGD_dropout_250.pth'
model_trans = build_UNet(type='UNet1', use_dropout=True, pretrained_model=pretrained_model)
if use_gpu:
    model_trans = model_trans.cuda()

criterion = nn.L1Loss()
optimizer_trans = optim.SGD(model_trans.parameters(), lr=lr, momentum=0.9, weight_decay=0)
train_model(model_trans, criterion, optimizer_trans, num_epochs=1)

