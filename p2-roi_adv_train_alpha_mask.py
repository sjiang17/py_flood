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
from roi_models import build_net, build_DNet
from roi_alpha_reader import FeatureReader

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_gpu = torch.cuda.is_available()
BATCH_SIZE = 16
DISPLAY = 500

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


def train_model(netG, netD, criterion_rec, criterion_adv, optimizer_trans, optimizer_D, num_epochs, dataloaders):
    since = time.time()
    
    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        summary_file = open(os.path.join(save_dir, 'log.txt'), 'a')
        summary_file.write("Epoch {} ".format(epoch))

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                # scheduler.step()
                netG.train(True)  # Set model to training mode
            else:
                netG.train(False)  # Set model to evaluate mode

            running_loss_trans = 0.0
            running_loss_adv_real = 0.0
            running_loss_adv_fake = 0.0
            running_loss_gen = 0.0
            running_loss_adv = 0.0

            acc_d1_iter, acc_d2_iter = 0.0, 0.0
            acc_d1_epoch, acc_d2_epoch = 0.0, 0.0

            # Iterate over data.
            for ix, data in enumerate(dataloaders[phase]):
                # get the inputs
                inputs, gt = data
                iter_batch_size = inputs.size(0)
                label_adv_tensor = torch.FloatTensor(iter_batch_size)  # BATCH_SIZE
                label_threshold = torch.ones(iter_batch_size) * 0.5

                # wrap them in Variable
                inputs = myGetVariable(inputs, use_gpu, phase)
                gt = myGetVariable(gt, use_gpu, phase)

                #################################
                # Update D network              #
                #################################

                #-------train with real-------
                optimizer_D.zero_grad()
                label_adv_real = myGetVariable(label_adv_tensor.fill_(1), use_gpu, phase)
                outputs_D_real = netD(gt)
                correct_r = (outputs_D_real.data.cpu().view(iter_batch_size) >= label_threshold).sum()
                # od1 = outputs_D_real.data.cpu()[0, 0]
                acc_d1_iter += correct_r 
                acc_d1_epoch += correct_r

                loss_D_real = criterion_adv(outputs_D_real, label_adv_real)
                if phase == 'train':
                    loss_D_real.backward()
                iter_loss_adv_real = loss_D_real.data[0] * inputs.size(0)

                #--------train with fake--------
                outputs_trans = netG(inputs)  # fake
                label_adv_fake = myGetVariable(label_adv_tensor.fill_(0), use_gpu, phase)
                outputs_D_fake = netD(outputs_trans.detach())
                # od2 = outputs_D_fake.data.cpu()[0, 0]
                correct_f = (outputs_D_fake.data.cpu().view(iter_batch_size) < label_threshold).sum()
                acc_d2_iter += correct_f
                acc_d2_epoch += correct_f

                loss_D_fake = criterion_adv(outputs_D_fake, label_adv_fake)
                if phase == 'train':
                    loss_D_fake.backward()
                    optimizer_D.step()
                iter_loss_adv_fake = loss_D_fake.data[0] * inputs.size(0)

                #################################
                # Update G network              #
                #################################

                # zero the parameter gradients
                optimizer_trans.zero_grad()

                # forward
                label_adv = myGetVariable(label_adv_tensor.fill_(1), use_gpu, phase)  # we want to generator to produce 1
                outputs_D = netD(outputs_trans)
                loss_gen = criterion_rec(outputs_trans, gt)
                loss_adv = criterion_adv(outputs_D, label_adv)
                loss_trans = loss_gen + lmda * loss_adv

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss_trans.backward()
                    optimizer_trans.step()

                iter_loss_trans = loss_trans.data[0] * inputs.size(0)
                iter_loss_gen = loss_gen.data[0] * inputs.size(0)
                iter_loss_adv = loss_adv.data[0] * inputs.size(0)

                if ix % DISPLAY == 0:
                    print('{}: iter {}, Loss Trans={:.4f}, Gen={:.4f}, Adv={:.4f}'.format(
                            datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), ix, 
                            iter_loss_trans, iter_loss_gen, iter_loss_adv))
                    print('{:18s}Adv Real={:.8f}, Adv Fake={:.8f}, acc real: {:.2f}, acc fake: {:.2f}'.format(
                          '', iter_loss_adv_real, iter_loss_adv_fake, acc_d1_iter/BATCH_SIZE / DISPLAY, acc_d2_iter/BATCH_SIZE / DISPLAY))
                    acc_d1_iter, acc_d2_iter = 0.0, 0.0

                # statistics
                running_loss_adv_real += iter_loss_adv_real
                running_loss_adv_fake += iter_loss_adv_fake
                running_loss_trans += iter_loss_trans
                running_loss_gen += iter_loss_gen
                running_loss_adv += iter_loss_adv

            epoch_loss_adv_real = running_loss_adv_real / dataset_sizes[phase]
            epoch_loss_adv_fake = running_loss_adv_fake / dataset_sizes[phase]
            epoch_loss_trans = running_loss_trans / dataset_sizes[phase]
            epoch_loss_gen = running_loss_gen / dataset_sizes[phase]
            epoch_loss_adv = running_loss_adv / dataset_sizes[phase]
            acc_d1_epoch /= dataset_sizes[phase]
            acc_d2_epoch /= dataset_sizes[phase]

            print('{} Loss Trans={:.4f}, Gen={:.4f}, Adv={:.4f}'.format(
                phase, epoch_loss_trans, epoch_loss_gen, epoch_loss_adv))
            print('Adv Real={:.8f}, Adv Fake={:.8f}, acc real={:.4f}, acc fake={:.4f}'.format(
                epoch_loss_adv_real, epoch_loss_adv_fake, acc_d1_epoch, acc_d2_epoch))

            summary_file.write("{}: {} Loss Trans={:.4f}, Gen={:.4f}, Adv={:.4f}\n".format(
                                datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), phase, 
                                epoch_loss_trans, epoch_loss_gen, epoch_loss_adv))
            summary_file.write('{:27s}Adv Real={:.8f}, Adv Fake={:.8f}, acc real={:.4f}, acc fake={:.4f}\n'.format(
                                '', epoch_loss_adv_real, epoch_loss_adv_fake, acc_d1_epoch, acc_d2_epoch))

            if epoch % 10 == 0:
                save_name_netG = 'netG_{}_{}.pth'.format(training_name, epoch)
                torch.save(netG.state_dict(), os.path.join(save_dir, save_name_netG))
                save_name_netD = 'netD_{}_{}.pth'.format(training_name, epoch)
                torch.save(netD.state_dict(), os.path.join(save_dir, save_name_netD))

        summary_file.write("\n")
        summary_file.close()
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return

############################
lr_g = 0.01
lr_d = 1e-3
lmda = 0.25
training_name = 'ADV_RoI_alpha_mask_kitti_lrg{}_lrd{}_lmda{}_bs{}_u2'.format(lr_g, lr_d, lmda, BATCH_SIZE)
# training_name = 'test3'

pairfile_dir = '/flpvc/dataset/kitti/vehicle/roi'
occ_data_dir = '/flpvc/dataset/kitti/vehicle/roi/occ/roi_feature'
unocc_data_dir = '/flpvc/dataset/kitti/vehicle/roi/unocc/roi_feature'
mask_data_dir = '/flpvc/dataset/kitti/vehicle/roi/mask/roi_feature'

featuremap_datasets = {x: FeatureReader(occ_data_dir, unocc_data_dir, mask_data_dir, pairfile_dir, 
			maker='make_dataset_mask', phase=x) for x in ['train', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(featuremap_datasets[x], batch_size=BATCH_SIZE,
                                              shuffle=False, num_workers=6)
                                              for x in ['train', 'test']}
dataset_sizes = {x: len(featuremap_datasets[x]) for x in ['train', 'test']}
print(dataset_sizes)

save_dir = os.path.join('save/adv', training_name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

pretrained_model = None
model_trans = build_net(type='RoI_UNet2', use_dropout=True, pretrained_model=pretrained_model)
model_D = build_DNet(pretrained_model=pretrained_model)

if use_gpu:
    model_trans = model_trans.cuda()
    model_D = model_D.cuda()

criterion_rec = nn.L1Loss()
criterion_adv = nn.BCELoss()
# optimizer_trans = optim.SGD(model_trans.parameters(), lr=lr, momentum=0.9, weight_decay=0.000)
optimizer_trans = optim.Adam(model_trans.parameters(), lr=lr_g, betas=(0.5, 0.999))
optimizer_D = optim.SGD(model_D.parameters(), lr=lr_d, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

print(training_name)
train_model(model_trans, model_D, criterion_rec, criterion_adv, optimizer_trans, optimizer_D, 400, dataloaders)
