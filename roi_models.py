import torch 
import torch.nn as nn
from torch.nn import init
			
class RoI_UNet(nn.Module):
	def __init__(self, use_bias=True, use_dropout=False):
		super(RoI_UNet, self).__init__()

		lrelu = nn.LeakyReLU(0.1, True)

		e1_conv = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias)
		e1_norm = nn.BatchNorm2d(512)
		self.e1 = nn.Sequential(e1_conv, e1_norm, lrelu)

		e2_conv = nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=use_bias)
		e2_norm = nn.BatchNorm2d(1024) 
		self.e2 = nn.Sequential(e2_conv, e2_norm, lrelu)
		
		e3_conv = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=use_bias)
		e3_norm = nn.BatchNorm2d(1024)
		self.e3 = nn.Sequential(e3_conv, e3_norm, lrelu)
		
		self.d1_deconv = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=use_bias)
		d1_norm = nn.BatchNorm2d(512)
		if use_dropout:
			self.d1 = nn.Sequential(d1_norm, nn.Dropout(0.5), lrelu)
		else:
			self.d1 = nn.Sequential(d1_norm, lrelu)
		
		d2_conv = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias)
		d2_norm = nn.BatchNorm2d(512)
		if use_dropout:
			self.d2 = nn.Sequential(d2_conv, d2_norm, nn.Dropout(0.5), lrelu)
		else:
			self.d2 = nn.Sequential(d2_conv, d2_norm, lrelu)

		d3_conv = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias)
		d3_relu = nn.ReLU(True)
		self.d3 = nn.Sequential(d3_conv, d3_relu)
		
		
	def forward(self, x):
		x_input	= x
		x = self.e1(x)
		x = self.e2(x)
		x = self.e3(x)
		x = self.d1_deconv(x, output_size=x_input.size())
		x = self.d1(x)
		x = self.d2(x)
		x = self.d3(x)
		
		return x

def weights_init_xavier(m):
	classname = m.__class__.__name__
	print(classname)
	if classname.find('Conv') != -1:
		init.xavier_normal(m.weight.data, gain=0.02)
	elif classname.find('Linear') != -1:
		init.xavier_normal(m.weight.data, gain=0.02)
	elif classname.find('BatchNorm2d') != -1:
		try: 
			init.normal(m.weight.data, 1.0, 0.02)
			init.constant(m.bias.data, 0.0)
		except:
			pass

def build_net(type='RoI_UNet', use_bias=True, use_dropout=False, pretrained_model=None):
	if type == 'RoI_UNet':
		model = RoI_UNet(use_bias=use_bias, use_dropout=use_dropout)
	else:
		print("No model type {}".format(type))
		exit()
	if pretrained_model == None:
		model.apply(weights_init_xavier)
	else: 
		model.load_state_dict(torch.load(pretrained_model))
	
	return model
	
        
      
