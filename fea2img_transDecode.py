from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from flood_models import build_UNet
from fea2img_model import build_model
from fea2img_reader import FmImgReader
import datetime
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def train_model(model, model_trans, criterion, optimizer, num_epochs):
    since = time.time()
    
    epoch = 1
    print('-' * 10)

    phase = 'test'

    model.train(False)  # Set model to evaluate mode
    model_trans.train(False)

    running_loss = 0.0

    # Iterate over data.
    for ix, data in enumerate(dataloaders[phase]):
        if ix == 300:
            break
	# get the fm
        fm, gt, base_name = data
        
        # wrap them in Variable
        if use_gpu:
            fm, gt = Variable(fm.cuda(), volatile=True), Variable(gt.cuda(), volatile=True)
        else:
            fm, gt = Variable(fm, volatile=True), Variable(gt, volatile=True)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        fm_trans = model_trans(fm)
        outputs = model(fm_trans)

        outputs_save_file = os.path.join(output_save_dir, base_name[0])
        img = transforms.ToPILImage()(outputs.data[0].cpu()).convert('RGB')
        img.save(outputs_save_file, format='JPEG')

        # loss = criterion(outputs, gt)
        loss = 0.0

        #iter_loss = loss.data[0] * fm.size(0)
        iter_loss = 0.0
        running_loss += iter_loss
        
        if ix % 100 == 0:
            print ('{}: iter {}, Loss = {:.4f}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), ix, iter_loss))

    epoch_loss = running_loss / dataset_sizes[phase]
    
    print('{}, {} Loss: {:.4f}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), phase, epoch_loss))
    print()
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return

lr = 0.01
# training_name = 'fm2img_conv4_lr{}_wd'.format(lr)
# training_name = 'test_conv3'

img_dir = '/pvdata/dataset/kitti/vehicle/mask_resize/'
#img_dir = '/pvdata/dataset/kitti/ped_inst/'
fm_dir = '/pvdata/dataset/kitti/vehicle/mask_resize/feature_map/feature_map-conv4pool/'
#fm_dir = '/pvdata/dataset/kitti/ped_inst/feature_map-conv4pool_resized/'

# Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] 
# to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
data_transforms = transforms.Compose([transforms.ToTensor()])

featuremap_datasets = {x: FmImgReader(os.path.join(img_dir, x, '0'),
					os.path.join(fm_dir,x, '0'), 
					transform=data_transforms)
                                          for x in ['test']}
dataloaders = {x: torch.utils.data.DataLoader(featuremap_datasets[x], batch_size=1,
                                                shuffle=False, num_workers=6)
                                                for x in ['test']}
dataset_sizes = {x: len(featuremap_datasets[x]) for x in ['test']}
print (dataset_sizes)

use_gpu = torch.cuda.is_available()

fea2img_pretrained_model = '/siyuvol/py_flood/save/fm2img_conv4_lr0.01_wd/transformer_fm2img_conv4_lr0.01_wd_400.pth'
output_save_dir = '/siyuvol/output/fea2img/all/ADVtransformed_conv4'
if not os.path.exists(output_save_dir):
    os.makedirs(output_save_dir)
    
model_fea2img = build_model(pretrained_model=fea2img_pretrained_model)

trans_pretrained_model = '/pvdata/savemodel/MASKED_kitti_adv_lrg0.001_lrd1e-06_lmda0.01_r3.0_SHALLOW_DROP/transformer_MASKED_kitti_adv_lrg0.001_lrd1e-06_lmda0.01_r3.0_SHALLOW_DROP_55.pth'
# trans_pretrained_model = '/siyuvol/py_flood/save/PEDINST_MASK_kitti_trans_lr0.01/transformer_PEDINST_MASK_kitti_trans_lr0.01_200.pth'
# trans_pretrained_model = '/pvdata/savemodel/MASK_kitti_UNet_lr0.01_SGD_dropout/transformer_MASK_kitti_UNet_lr0.01_SGD_dropout_250.pth'
model_trans = build_UNet(type='UNet1', use_dropout=False, pretrained_model=trans_pretrained_model)

if use_gpu:
    model_fea2img = model_fea2img.cuda()
    model_trans = model_trans.cuda()

criterion = nn.L1Loss()
optimizer_trans = optim.SGD(model_trans.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
# optimizer_trans = optim.Adam(model_trans.parameters(), lr=lr, weight_decay=0.0005)

# Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

print(fea2img_pretrained_model)
train_model(model_fea2img, model_trans, criterion, optimizer_trans, num_epochs=400)
