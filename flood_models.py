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

class UNet_conv3(nn.Module):
	def __init__(self, in_channels=256, use_bias=True, use_dropout=True):
		super(UNet_conv3, self).__init__()
		
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
		d1_relu = nn.LeakyReLU(True)
		if use_dropout:
			self.d1 = nn.Sequential(d1_norm, nn.Dropout(0.5), d1_relu)
		else:
			self.d1 = nn.Sequential(d1_norm, d1_relu)
		
		self.d2_deconv = nn.ConvTranspose2d(2048, 512, kernel_size=4, stride=2, padding=1, bias=use_bias)
		d2_norm = nn.BatchNorm2d(512)
		d2_relu = nn.LeakyReLU(True)
		if use_dropout:
			self.d2 = nn.Sequential(d2_norm, nn.Dropout(0.5), d2_relu)
		else:
			self.d2 = nn.Sequential(d2_norm, d2_relu)
		
		self.d3_deconv = nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1, bias=use_bias)
		d3_norm = nn.BatchNorm2d(256)
		d3_relu = nn.LeakyReLU(True)
		if use_dropout:
			self.d3 = nn.Sequential(d3_norm, nn.Dropout(0.5), d3_relu)
		else:
			self.d3 = nn.Sequential(d3_norm, d3_relu)
		
		d4_conv = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
		d4_relu = nn.ReLU(True)
		self.d4 = nn.Sequential(d4_conv, d4_relu)
		
		
	def forward(self, x):
		x_input	= x
		x_e1 = self.e1(x_input)
		x_e2 = self.e2(x_e1)
		x = self.e3(x_e2)
		x = self.d1_deconv(x, output_size=x_e2.size())
		x = self.d1(x)
		x = self.d2_deconv(torch.cat([x_e2, x], 1), output_size=x_e1.size())
		x = self.d2(x)
		x = self.d3_deconv(torch.cat([x_e1, x], 1), output_size=x_input.size())
		x = self.d3(x)
		x = self.d4(torch.cat([x_input, x], 1))
		
		return x

class UNet_conv3_d(nn.Module):
	def __init__(self, in_channels=256, use_bias=True, use_dropout=True):
		super(UNet_conv3_d, self).__init__()

		e1_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=use_bias)
		e1_norm = nn.BatchNorm2d(in_channels)
		e1_relu = nn.LeakyReLU(0.2, True)
		self.e1 = nn.Sequential(e1_conv, e1_norm, e1_relu)
		
		e2_conv = nn.Conv2d(in_channels, 512, kernel_size=4, stride=2, padding=1, bias=use_bias)
		e2_norm = nn.BatchNorm2d(512) # inner_nc
		e2_relu = nn.LeakyReLU(0.2, True)
		self.e2 = nn.Sequential(e2_conv, e2_norm, e2_relu)
		
		e3_conv = nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=use_bias)
		e3_norm = nn.BatchNorm2d(1024)
		e3_relu = nn.LeakyReLU(0.2, True)
		self.e3 = nn.Sequential(e3_conv, e3_norm, e3_relu)
		
		e4_conv = nn.Conv2d(1024, 2048, kernel_size=4, stride=2, padding=1, bias=use_bias)
		e4_norm = nn.BatchNorm2d(2048)
		e4_relu = nn.LeakyReLU(0.2, True)
		self.e4 = nn.Sequential(e4_conv, e4_norm, e4_relu)

		e5_conv = nn.Conv2d(2048, 4096, kernel_size=4, stride=2, padding=1, bias=use_bias)
		e5_norm = nn.BatchNorm2d(4096)
		e5_relu = nn.LeakyReLU(0.2, True)
		self.e5 = nn.Sequential(e5_conv, e5_norm, e5_relu)

		self.d1_deconv = nn.ConvTranspose2d(4096, 2048, kernel_size=4, stride=2, padding=1, bias=use_bias)
		d1_norm = nn.BatchNorm2d(2048)
		d1_relu = nn.LeakyReLU(True)
		if use_dropout:
			self.d1 = nn.Sequential(d1_norm, nn.Dropout(0.5), d1_relu)
		else:
			self.d1 = nn.Sequential(d1_norm, d1_relu)
		
		self.d2_deconv = nn.ConvTranspose2d(4096, 1024, kernel_size=4, stride=2, padding=1, bias=use_bias)
		d2_norm = nn.BatchNorm2d(1024)
		d2_relu = nn.LeakyReLU(True)
		if use_dropout:
			self.d2 = nn.Sequential(d2_norm, nn.Dropout(0.5), d2_relu)
		else:
			self.d2 = nn.Sequential(d2_norm, d2_relu)
		
		self.d3_deconv = nn.ConvTranspose2d(2048, 512, kernel_size=4, stride=2, padding=1, bias=use_bias)
		d3_norm = nn.BatchNorm2d(512)
		d3_relu = nn.LeakyReLU(True)
		if use_dropout:
			self.d3 = nn.Sequential(d3_norm, nn.Dropout(0.5), d3_relu)
		else:
			self.d3 = nn.Sequential(d3_norm, d3_relu)
		
		self.d4_deconv = nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1, bias=use_bias)
		d4_norm = nn.BatchNorm2d(256)
		d4_relu = nn.LeakyReLU(True)
		if use_dropout:
			self.d4 = nn.Sequential(d4_norm, nn.Dropout(0.5), d4_relu)
		else:
			self.d4 = nn.Sequential(d4_norm, d4_relu)
		
		d5_conv = nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=use_bias)
		d5_relu = nn.ReLU(True)
		self.d5 = nn.Sequential(d5_conv, d5_relu)
		
		d6_conv = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=use_bias)
		d6_relu = nn.ReLU(True)
		self.d6 = nn.Sequential(d6_conv, d6_relu)
		
	def forward(self, x):
		x_input	= x
		x_e1 = self.e1(x_input)
		x_e2 = self.e2(x_e1)
		x_e3 = self.e3(x_e2)
		x_e4 = self.e4(x_e3)
		x = self.e5(x_e4)
		x = self.d1_deconv(x, output_size=x_e4.size())
		x = self.d1(x)
		x = self.d2_deconv(torch.cat([x_e4, x], 1), output_size=x_e3.size())
		x = self.d2(x)
		x = self.d3_deconv(torch.cat([x_e3, x], 1), output_size=x_e2.size())
		x = self.d3(x)
		x = self.d4_deconv(torch.cat([x_e2, x], 1), output_size=x_input.size())
		x = self.d4(x)
		x = self.d5(torch.cat([x_input, x], 1))
		x = self.d6(x)
		
		return x

class UNet_three2one(nn.Module):
	def __init__(self, in_channels=256, use_bias=True, use_dropout=True):
		super(UNet_three2one, self).__init__()

		lrelu = nn.LeakyReLU(0.1, True)

		conv3_1 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=use_bias)
		conv3_2 = nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=use_bias)
		self.conv3 = nn.Sequential(conv3_1, nn.BatchNorm2d(512), lrelu,
									conv3_2, nn.BatchNorm2d(1024, affine=False), lrelu)

		conv4_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=use_bias)
		conv4_2 = nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=use_bias)
		self.conv4 = nn.Sequential(conv4_1, nn.BatchNorm2d(512), lrelu,
									conv4_2, nn.BatchNorm2d(1024, affine=False), lrelu)

		conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=use_bias)
		conv5_2 = nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=use_bias)
		self.conv5 = nn.Sequential(conv5_1, nn.BatchNorm2d(512), lrelu,
									conv5_2, nn.BatchNorm2d(1204, affine=False), lrelu)
		# 32x
		e1_conv = nn.Conv2d(1024, 1024, kernel_size=3, padding=1, bias=use_bias)
		self.e1 = nn.Sequential(e1_conv, nn.BatchNorm2d(1024), lrelu)
		
		# 64x
		e2_conv = nn.Conv2d(1024, 2048, kernel_size=4, stride=2, padding=1, bias=use_bias)
		self.e2 = nn.Sequential(e2_conv, nn.BatchNorm2d(2048), lrelu)
		
		# 128x
		e3_conv = nn.Conv2d(2048, 4096, kernel_size=4, stride=2, padding=1, bias=use_bias)
		self.e3 = nn.Sequential(e3_conv, nn.BatchNorm2d(4096), lrelu)
		
		# e4_conv = nn.Conv2d(4096, 4096, kernel_size=3, padding=1, bias=use_bias)
		# self.e4 = nn.Sequential(e4_conv, nn.BatchNorm2d(4096), lrelu)

		self.d1_deconv = nn.ConvTranspose2d(4096, 2048, kernel_size=4, stride=2, padding=1, bias=use_bias)
		if use_dropout:
			self.d1 = nn.Sequential(nn.BatchNorm2d(2048), nn.Dropout(0.5), lrelu)
		else:
			self.d1 = nn.Sequential(nn.BatchNorm2d(2048), lrelu)
		
		self.d2_deconv = nn.ConvTranspose2d(4096, 1024, kernel_size=4, stride=2, padding=1, bias=use_bias)
		if use_dropout:
			self.d2 = nn.Sequential(nn.BatchNorm2d(1024), nn.Dropout(0.5), lrelu)
		else:
			self.d2 = nn.Sequential(nn.BatchNorm2d(1024), lrelu)
		
		d3_conv = nn.Conv2d(2048, 1024, kernel_size=3, padding=1, bias=use_bias)
		if use_dropout:
			self.d3 = nn.Sequential(d3_conv, nn.BatchNorm2d(1024), nn.Dropout(0.5), lrelu)
		else:
			self.d3 = nn.Sequential(d3_conv, nn.BatchNorm2d(1024), lrelu)
		
		d4_conv = nn.Conv2d(1024, 1024, kernel_size=3, padding=1, bias=use_bias)
		if use_dropout:
			self.d4 = nn.Sequential(d4_conv, nn.BatchNorm2d(1024), nn.Dropout(0.5), lrelu)
		else:
			self.d4 = nn.Sequential(d4_conv, nn.BatchNorm2d(1024), lrelu)

		d5_conv = nn.Conv2d(1024, 1024, kernel_size=3, padding=1, bias=use_bias)
		self.d5 = nn.Sequential(d5_conv, nn.ReLU(True))

	def forward(self, x_conv3, x_conv4, x_conv5):
		x_o3 = self.conv3(x_conv3)
		x_o4 = self.conv4(x_conv4)
		x_o5 = self.conv5(x_conv5)
		x_o = x_o3 + x_o5 + x_o5

		x_e1 = self.e1(x_o)
		x_e2 = self.e2(x_e1)
		x = self.e3(x_e2)
		x = self.d1_deconv(x, output_size=x_e2.size())
		x = self.d1(x)
		x = self.d2_deconv(torch.cat([x_e2, x], 1), output_size=x_conv5.size())
		x = self.d3(torch.cat([x_e1, x], 1))
		x = self.d4(x)
		x = self.d5(x)
		
		return x


def weights_init_xavier(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		init.xavier_normal(m.weight.data, gain=0.02)
	elif classname.find('Linear') != -1:
		init.xavier_normal(m.weight.data, gain=0.02)
	elif classname.find('BatchNorm2d') != -1:
		init.normal(m.weight.data, 1.0, 0.02)
		init.constant(m.bias.data, 0.0)

def build_UNet(type='UNet1', use_bias=True, use_dropout=False, pretrained_model=None):
	if type == 'UNet1':
		model = UNet(use_bias=use_bias, use_dropout=use_dropout)
	elif type == 'UNet2':
		model = UNet2()
	elif type == 'UNet1_Conv3':
		model = UNet_conv3(use_bias=use_bias, use_dropout=use_dropout)
	elif type == 'UNet1_conv3_d':
		model = UNet_conv3_d(use_bias=use_bias, use_dropout=use_dropout)
	else:
		print("No model type {}".format(type))
		exit()
	if pretrained_model == None:
		model.apply(weights_init_xavier)
	else: 
		model.load_state_dict(torch.load(pretrained_model))
			
	return model
	
        
      
