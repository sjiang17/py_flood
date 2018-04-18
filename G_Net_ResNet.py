import torch
import torch.nn as nn
from torch.nn import init
import functools
import numpy as np

# modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
# siyu 2018-03-14
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=512, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=5, gpu_ids=[], padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        # i = 0, 1, mult = 1, 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
        # mult = 4
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        
        self.e = nn.Sequential(*model)
        
        # i = 0, 1, mult = 1/2, 1
        # for i in range(n_downsampling):
        #     mult = 2**(n_downsampling - i)
        #     model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
        #                                  kernel_size=3, stride=2,
        #                                  padding=1, output_padding=1,
        #                                  bias=use_bias),
        #               norm_layer(int(ngf * mult / 2)),
        #               nn.ReLU(True)]
                      
        i = 0
        mult = 2**(n_downsampling - i)
        self.d1_deconv = nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=4, stride=2,
                                         padding=1, bias=use_bias)
        self.d1 = nn.Sequential(
        	norm_layer(int(ngf * mult / 2)),
        	nn.ReLU(True)
        	)
        
        i = 1
        mult = 2**(n_downsampling - i)
        self.d2_deconv = nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=4, stride=2,
                                         padding=1, bias=use_bias)
        self.d2 = nn.Sequential(
        	norm_layer(int(ngf * mult / 2)),
        	nn.ReLU(True)
        	)
        
        # model += [nn.ReflectionPad2d(3)]
        # model += [nn.Conv2d(ngf, output_nc, kernel_size=3, padding=1)]
        # model += [nn.Tanh()]
        # model += [nn.ReLU(True)]
        self.d3 = nn.Sequential(
        	nn.Conv2d(ngf, output_nc, kernel_size=3, padding=1),
        	nn.ReLU(True)
        	)

        # self.model = nn.Sequential(*model)

    def forward(self, x):
        # if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
        #     return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        # else:
        #     return self.model(input)
        
        # h/2 w/2 when dosample once
        s1 = x.size()
        s2 = np.array(s1)
        s2[2] = s1[2] / 2
        s2[3] = s1[3] / 2

        x = self.e(x)
        x = self.d1_deconv(x, output_size=s2)
        x = self.d1(x)
        x = self.d2_deconv(x, output_size=s1)
        x = self.d2(x)
        x = self.d3(x)
        return x
        

# class FixedTransposeConv(nn.Module):
# 	def __init__(self, in_channel, out_channel, kernel_size=4, stride=2, padding=1, use_bias=False):
# 		super(FixedTransposeConv, self).__init__()
# 		self.transpose_conv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=use_bias)
		
		
# 	def forward(self, x):
# 		x = self.transpose_conv(x, output_size=self.output_size)
# 		return x


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_xavier(m):
	classname = m.__class__.__name__
	# print(classname)
	if classname.find('Conv') != -1:
		init.xavier_normal(m.weight.data, gain=0.02)
	elif classname.find('Linear') != -1:
		init.xavier_normal(m.weight.data, gain=0.02)
	elif classname.find('BatchNorm2d') != -1:
		init.normal(m.weight.data, 1.0, 0.02)
		init.constant(m.bias.data, 0.0)
		
def build_GNet_ResNet(use_dropout):
    input_nc = 512
    output_nc = 512
    ngf = 512   
    residual_blocks = 5
    norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    model = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=residual_blocks)
    model.apply(weights_init_xavier)
    return model