import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torch

class VGG_tr(nn.Module):

    def __init__(self, features, num_classes=2, init_weights=True, defualt_input_size=512):
        super(VGG_tr, self).__init__()
        self.RCNN_base = features
        # self.classifier = nn.Sequential(
        #     nn.Linear(512 * (defualt_input_size/32) * (defualt_input_size/32), 1024),
        #     nn.ReLU(True),
        #     # nn.Dropout(),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(True),
        #     # nn.Dropout(),
        #     nn.Linear(1024, num_classes),
        # )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.RCNN_base(x)
        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'M4':
            layers += [nn.MaxPool2d(kernel_size=4, stride=4)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'CONV4': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M'],
    'CONV4M4': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M4'],
    'CONV4NoM': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512],
    'CONV3': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M'],
    'CONV5': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
}

def vgg_tr_conv4(pretrained=False, **kwargs):
	
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_tr(make_layers(cfg['CONV4']), **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
        # pretrained_dict = model_zoo.load_url(model_urls['vgg16'])
        pretrianed_model_dir = '/siyuvol/pytorch-model/vgg16/pascal_voc/faster_rcnn_1_10_625.pth'
        pretrained_dict = torch.load(pretrianed_model_dir)['model']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        for kk in pretrained_dict:
            print kk
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

def vgg_tr_conv3(pretrained=False, **kwargs):
    
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_tr(make_layers(cfg['CONV3']), **kwargs)
    if pretrained:
        pretrianed_model_dir = '/siyuvol/pytorch-model/vgg16/pascal_voc/faster_rcnn_1_10_625.pth'
        pretrained_dict = torch.load(pretrianed_model_dir)['model']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        for kk in pretrained_dict:
            print kk
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

def vgg_tr_conv5(pretrained=False, **kwargs):
    
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_tr(make_layers(cfg['CONV5']), **kwargs)
    if pretrained:
        pretrianed_model_dir = '/siyuvol/pytorch-model/vgg16/pascal_voc/faster_rcnn_1_10_625.pth'
        pretrained_dict = torch.load(pretrianed_model_dir)['model']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        for kk in pretrained_dict:
            print kk
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model
