import torch 
import torch.nn as nn
from torch.nn import init

class UNet(nn.Module):
	def __init__(self, in_channels=512, use_bias=True, use_dropout=True):
		super(UNet, self).__init__()
		
		#					input_nc,   inner_nc
		e1_conv = nn.Conv2d(in_channels, 512, kernel_size=4, stride=2, padding=1, bias=use_bias)
		e1_norm = nn.BatchNorm2d(512) # inner_nc
		e1_relu = nn.LeakyReLU(0.2, True)
		self.e1 = nn.Sequential(e1_conv, e1_norm, e1_relu)
		
		e2_conv = nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=use_bias)
		e2_norm = nn.BatchNorm2d(1024)
		e2_relu = nn.LeakyReLU(0.2, True)
		self.e2 = nn.Sequential(e2_conv, e2_norm, e2_relu)
		
		e3_conv = nn.Conv2d(1024, 2048, kernel_size=4, stride=2, padding=1, bias=use_bias)
		# e3_norm = nn.BatchNorm2d(2048)
		e3_relu = nn.LeakyReLU(0.2, True)
		self.e3 = nn.Sequential(e3_conv, e3_relu)
		
		self.d1_deconv = nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1, bias=use_bias)
		# d1_deconv_pad = d1_deconv(output_size=10)
		d1_norm = nn.BatchNorm2d(1024)
		d1_relu = nn.ReLU(True)
		if use_dropout:
			self.d1 = nn.Sequential(d1_norm, nn.Dropout(0.5), d1_relu)
		else:
			self.d1 = nn.Sequential(d1_norm, d1_relu)
		
		self.d2_deconv = nn.ConvTranspose2d(2048, 512, kernel_size=4, stride=2, padding=1, bias=use_bias)
		d2_norm = nn.BatchNorm2d(512)
		d2_relu = nn.ReLU(True)
		if use_dropout:
			self.d2 = nn.Sequential(d2_norm, nn.Dropout(0.5), d2_relu)
		else:
			self.d2 = nn.Sequential(d2_norm, d2_relu)
		
		self.d3_deconv = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=use_bias)
		d3_norm = nn.BatchNorm2d(512)
		d3_relu = nn.ReLU(True)
		if use_dropout:
			self.d3 = nn.Sequential(d3_norm, nn.Dropout(0.5), d3_relu)
		else:
			self.d3 = nn.Sequential(d3_norm, d3_relu)
		
		d4_conv = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1)
		d4_relu = nn.ReLU(True)
		# d4_norm = nn.BatchNorm2d(512)
		# d4_relu = nn.ReLU(True)
		# if use_dropout:
		# 	self.d4 = nn.Sequential(d4_deconv, d4_norm, nn.Dropout(0.5), d4_relu)
		# else:
		# 	self.d4 = nn.Sequential(d4_deconv, d4_norm, d4_relu)
		self.d4 = nn.Sequential(d4_conv, d4_relu)
		
		
	def forward(self, x):
		x_input	= x
		x_e1 = self.e1(x_input)
		x_e2 = self.e2(x_e1)
		# print x_e2.size()
		x = self.e3(x_e2)
		x = self.d1_deconv(x, output_size=x_e2.size())
		x = self.d1(x)
		x = self.d2_deconv(torch.cat([x_e2, x], 1), output_size=x_e1.size())
		x = self.d2(x)
		x = self.d3_deconv(torch.cat([x_e1, x], 1), output_size=x_input.size())
		x = self.d3(x)
		x = self.d4(torch.cat([x_input, x], 1))
		
		return x
			
	# def _initialize_weights(self):
class UNet2(nn.Module):
	def __init__(self, in_channels=512, use_bias=True, use_dropout=False):
		super(UNet2, self).__init__()
		# p1 = nn.MaxPool2d(kernel_size=2, stride=2)
		
		#					input_nc,   inner_nc
		self.e1 = nn.Sequential(
			nn.Conv2d(in_channels, 512, kernel_size=4, stride=2, padding=1, bias=use_bias),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.2, True),
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.2, True),
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.2, True)
			)
		
		self.e2 = nn.Sequential(
			nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=use_bias),
			nn.BatchNorm2d(1024),
			nn.LeakyReLU(0.2, True),
			nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=use_bias),
			nn.BatchNorm2d(1024),
			nn.LeakyReLU(0.2, True),
			nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=use_bias),
			nn.BatchNorm2d(1024),
			nn.LeakyReLU(0.2, True)
			)
		
		self.e3 = nn.Sequential(
			nn.Conv2d(1024, 2048, kernel_size=4, stride=2, padding=1, bias=use_bias),
			nn.LeakyReLU(0.2, True),
			nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=1, bias=use_bias),
			nn.LeakyReLU(0.2, True),
			)
		
		self.d1_deconv = nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1, bias=use_bias)
		self.d1_conv = nn.Sequential(
			nn.BatchNorm2d(1024),
			nn.ReLU(True),
			nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(1024),
			nn.ReLU(True),
			nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(1024),
			nn.ReLU(True),
			)
		
		self.d2_deconv = nn.ConvTranspose2d(2048, 512, kernel_size=4, stride=2, padding=1, bias=use_bias)		
		self.d2_conv = nn.Sequential(
			nn.BatchNorm2d(512),
			nn.ReLU(True),
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(True),
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(True),
			)
		
		self.d3_deconv = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=use_bias)
		self.d3_conv = nn.Sequential(
			nn.BatchNorm2d(512),
			nn.ReLU(True),
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(True),
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(True),
			)
		
		self.d4 = nn.Sequential(
			nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(True),
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(True),
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
			nn.ReLU(True)
			)
		
	def forward(self, x):
		x_input	= x
		x_e1 = self.e1(x_input)
		x_e2 = self.e2(x_e1)
		# print x_e2.size()
		x = self.e3(x_e2)
		x = self.d1_deconv(x, output_size=x_e2.size())
		x = self.d1_conv(x)
		x = self.d2_deconv(torch.cat([x_e2, x], 1), output_size=x_e1.size())
		x = self.d2_conv(x)
		x = self.d3_deconv(torch.cat([x_e1, x], 1), output_size=x_input.size())
		x = self.d3_conv(x)
		x = self.d4(torch.cat([x_input, x], 1))
		
		return x

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

def build_UNet(type='UNet1', use_bias=True, use_dropout=False, is_pretrained=False):
	if type == 'UNet1':
		model = UNet(use_bias=use_bias, use_dropout=use_dropout)
	else:
		model = UNet2()
	if not is_pretrained:
		model.apply(weights_init_xavier)
	else: 
		# pretrained_dict = torch.load('/home/tmu/py_flood/save/conv4_lr0.01/transformer_conv4_lr0.01_24.pth')
		# for kk in pretrained_dict:
		# 	print kk
		# exit()
		if type == 'UNet1':
			model.load_state_dict(torch.load('/siyuvol/py_flood/save/MASK_kitti_UNet_lr0.01_SGD_dropout/transformer_MASK_kitti_UNet_lr0.01_SGD_dropout_350.pth'))
		elif type == 'UNet2':
			model.load_state_dict(torch.load(''))
	return model
	
        
      