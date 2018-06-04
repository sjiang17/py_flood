from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
# from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms
import os
import time
import datetime
from roi_models import build_net
from roi_reader import FeatureReader

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
use_gpu = torch.cuda.is_available()

def train_model(model, criterion, optimizer, num_epochs, dataloaders):
    since = time.time()
    
    for epoch in range(1, num_epochs+1):
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
                loss = criterion(outputs, gt)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                
                iter_loss = loss.data[0] * inputs.size(0)
                running_loss += iter_loss
                
                if ix % 1000 == 0:
                    print ('{}: iter {}, Loss = {:.4f}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), ix, iter_loss))

            epoch_loss = running_loss / dataset_sizes[phase]
            
            print('{}, {} Loss: {:.4f}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), phase, epoch_loss))
            summary_file.write("{}, {} loss {:.4f} ".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), phase, epoch_loss))
            
            if epoch == 1 or epoch % 10 == 0:
                save_name = '{}_{}.pth'.format(training_name ,epoch)
                torch.save(model.state_dict(), os.path.join(save_dir, save_name))
        
        summary_file.write("\n")
        summary_file.close()
        print()
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return

############################
lr = 0.01
training_name = 'RoI_kitti_lr{}_wd'.format(lr)
# training_name = 'test3'

pairfile_dir = '/pvdata/dataset/kitti/vehicle/roi'
occ_data_dir = '/pvdata/dataset/kitti/vehicle/roi/occ/roi_feature'
unocc_data_dir = '/pvdata/dataset/kitti/vehicle/roi/unocc/roi_feature'

featuremap_datasets = {x: FeatureReader(occ_data_dir, unocc_data_dir, pairfile_dir, phase=x)
                                          for x in ['train', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(featuremap_datasets[x], batch_size=1,
                                                shuffle=False, num_workers=6)
                                                for x in ['train', 'test']}
dataset_sizes = {x: len(featuremap_datasets[x]) for x in ['train', 'test']}
print(dataset_sizes)

save_dir = os.path.join('save', training_name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

pretrained_model = None
model_trans = build_net(type='RoI_UNet', use_dropout=True, pretrained_model=pretrained_model)
if use_gpu:
    model_trans = model_trans.cuda()

criterion = nn.L1Loss()
optimizer_trans = optim.SGD(model_trans.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
# optimizer_trans = optim.Adam(model_trans.parameters(), lr=lr, weight_decay=0.0005)

# Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

print(training_name)
train_model(model_trans, criterion, optimizer_trans, 400, dataloaders)
