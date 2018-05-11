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
from read_featuremap_nnsearch import FeatureReader
import cPickle

def get_Variable(var):
    if use_gpu:
        var = Variable(var.cuda(), volatile=True)
    else:
        var = Variable(var, volatile=True)
    return var

def nn_search(model, criterion, optimizer, num_epochs=1):
    since = time.time()
    neighbor_vec = np.zeros((dataset_sizes, dataset_sizes))
    model.train(False)
    
    for ixb, base_data in enumerate(base_dataloaders):
    	if ixb == 300:
            break
        base_fm, ind_b = base_data
    	print ("base_data {} {}".format(ixb, ind_b[0]))
    	summary_file = open(os.path.join(save_dir, 'log.txt'), 'a')
        summary_file.write("{} base_data {} {}:".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), ixb, ind_b[0]))
	
        base_fm = get_Variable(base_fm)
        outputs = model(base_fm)
        
        # Iterate over data.
        for ix, data in enumerate(target_dataloaders):
            # get the inputs
            target_fm, target_name = data
            # wrap them in Variable
            target_fm = get_Variable(target_fm)
                        # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            loss = criterion(outputs, target_fm)

            iter_loss = loss.data[0] #* base_fm.size(0)
            neighbor_vec[ixb, ix] = iter_loss
            if ix % 500 == 0:
                print ('{} iter {}, Loss = {:.4f}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), ix, iter_loss))

        nearests = np.argsort(neighbor_vec[ixb])[:5]
        for n in list(nearests):
        	# print ("{} {}".format(n, neighbor_vec[ixb, n]))
        	summary_file.write(" {} {:.4f},".format(n, neighbor_vec[ixb, n]))
        summary_file.write("\n\n")
        summary_file.close()
        print()
   
    pickleFile = os.path.join(save_dir, 'MASK_kitti_UNet_lr0.01_SGD_dropout_250_nn.pkl')
    cPickle.dump(neighbor_vec, open(pickleFile, 'w'))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
use_gpu = torch.cuda.is_available()

base_data_dir = '/pvdata/dataset/kitti/vehicle/mask_resize/feature_map-conv4pool/test/1'
target_data_dir = '/pvdata/dataset/kitti/vehicle/mask_resize/feature_map-conv4pool/test/0'

target_datasets = FeatureReader(target_data_dir)
dataset_sizes = len(target_datasets)
target_dataloaders = torch.utils.data.DataLoader(target_datasets, batch_size=1,
                                                shuffle=False, num_workers=4)

base_datasets = FeatureReader(base_data_dir)
assert dataset_sizes == len(base_datasets)
print(dataset_sizes)
base_dataloaders = torch.utils.data.DataLoader(base_datasets, batch_size=1,
                                                shuffle=False, num_workers=4)

save_dir = os.path.join('save', 'nnsearch_MASK_kitti_UNet_lr0.01_SGD_dropout_250')
if not os.path.exists(save_dir):
	os.makedirs(save_dir)
pretrained_model = '/pvdata/savemodel/MASK_kitti_UNet_lr0.01_SGD_dropout/transformer_MASK_kitti_UNet_lr0.01_SGD_dropout_250.pth'
model_trans = build_UNet(type='UNet1', use_dropout=True, pretrained_model=pretrained_model)
if use_gpu:
    model_trans = model_trans.cuda()

criterion = nn.L1Loss()
optimizer_trans = optim.SGD(model_trans.parameters(), lr=0.01, momentum=0.9, weight_decay=0)
nn_search(model_trans, criterion, optimizer_trans, num_epochs=1)

