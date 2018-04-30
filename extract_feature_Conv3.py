from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
# from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, transforms
import time
import os
import copy
from truncated_vgg import vgg_tr_conv3
# try:
# 	import cPickle as pickle 
# except:
# 	import pickle
import h5py
from image_reader import ImageFolder

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data_transforms = {
    'train': transforms.Compose([
        # transforms.RandomSizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        #Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] 
        # to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        # transforms.Scale(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


data_dir = '/siyuvol/dataset/kitti/greymask/train/0'
image_datasets = ImageFolder(data_dir, data_transforms['test'])
dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=1,
                                           shuffle=False, num_workers=4)
dataset_sizes = len(image_datasets)

use_gpu = torch.cuda.is_available()

save_dir = '/siyuvol/dataset/kitti/greymask/feature_map-conv3pool/train/0'
if not os.path.exists(save_dir):
	os.makedirs(save_dir)

def extract_model(model, criterion=None, optimizer=None):
    since = time.time()

    # Each epoch has a training and validation phase
    phase = 'test'
    model.train(False)  # Set model to evaluate mode

    # Iterate over data.
    for ix, data in enumerate(dataloaders):
        # get the inputs
        inputs, fbasename = data
        
        # wrap them in Variable
        if use_gpu:
            inputs = Variable(inputs.cuda(), volatile=True)
        else:
            inputs = Variable(inputs, volatile=True)

        # forward
        outputs = model(inputs)
        # print (outputs.shape)
        
        save_file = os.path.join(save_dir, fbasename[0]+'.h5')
        
        h5_file = h5py.File(save_file,'w')
        h5_file['data'] = outputs.data[0,:,:,:]
        h5_file.close()
        
        if ix % 200 == 0:
        	print ("finished {} out of {} images".format(ix, dataset_sizes))

    time_elapsed = time.time() - since
    print('Extraction completed in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    
    return 


model_ft = vgg_tr_conv3(pretrained=True, defualt_input_size=600)
if use_gpu:
    model_ft = model_ft.cuda()

model_ft = extract_model(model_ft)
