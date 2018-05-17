import torch 
import torch.nn as nn
from torch.nn import init

class DecoderNet(nn.Module):
	def __init__(self, use_bias=True):
		super(DecoderNet, self).__init__()

		lrelu = nn.LeakyReLU(0.1, True)

		self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, bias=use_bias)
		conv1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=use_bias)
		self.d1 = nn.Sequential(nn.BatchNorm2d(512), lrelu,
								conv1, nn.BatchNorm2d(512), lrelu,
								conv1, nn.BatchNorm2d(512), lrelu)

		self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=use_bias)
		conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=use_bias)
		self.d2 = nn.Sequential(nn.BatchNorm2d(256), lrelu,
								conv2, nn.BatchNorm2d(256), lrelu,
								conv2, nn.BatchNorm2d(256), lrelu)

		self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=use_bias)
		conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=use_bias)
		self.d3 = nn.Sequential(nn.BatchNorm2d(128), lrelu,
								conv3, nn.BatchNorm2d(128), lrelu)

		self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=use_bias)
		conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=use_bias)
		self.d4 = nn.Sequential(nn.BatchNorm2d(64), lrelu,
								conv3, nn.BatchNorm2d(64), lrelu)

		conv5_1 = nn.Conv2d(64, 3, kernel_size=3, padding=1, bias=use_bias)
		conv5_2 = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=use_bias)
		self.conv5 = nn.Sequential(conv5_1, nn.BatchNorm2d(3), lrelu,
									conv5_2, nn.BatchNorm2d(3), lrelu,
									conv5_2, nn.Tanh())

	def forward(self, x):
		img_size = (600, 1000)
		x = self.deconv1(x)
		x = self.d1(x)
		x = self.deconv2(x)
		x = self.d2(x)
		x = self.deconv3(x)
		x = self.deconv4(x, output_size=img_size)
		x = self.conv5(x)
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

def build_model(use_bias=True, pretrained_model=None):
	model = DecoderNet(use_bias=use_bias)
	if pretrained_model == None:
		model.apply(weights_init_xavier)
	else: 
		model.load_state_dict(torch.load(pretrained_model))
	return model





