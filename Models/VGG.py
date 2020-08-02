import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import json

import torch.nn.functional as Function

from Modules import ACT_Q, Linear_Q, ACT_fq
from Modules.Conv2d_quan import QuantConv2d as Conv2d_quan

import json
from collections import OrderedDict

cfg = {
    'VGG7': [128, 128, 'M', 256, 256, 'M', 512, 512, 'M'],
    'VGG7Q': [128, 128, 'M', 256, 256, 'M', 512],
}

__all__ = [
     'cifar10_vggsmall_qfnv2' , 'cifar10_vggsmall_q'
]

model_urls = {
    'vgg7': 'xxx',
}

def load_fake_quantized_state_dict(model, original_state_dict, key_map=None):
    original_state_dict = original_state_dict['state_dict']
    if not isinstance(key_map, OrderedDict):
        with open('{}'.format(key_map)) as rf:
            key_map = json.load(rf)
    for k, v in key_map.items():
        if 'num_batches_tracked' in k:
            continue
        if 'expand_' in k and model.state_dict()[k].shape != original_state_dict[v].shape:
            ori_weight = original_state_dict[v]
            new_weight = torch.cat((ori_weight, ori_weight * 2 ** 4), dim=1)
            model.state_dict()[k].copy_(new_weight)
        else:
            model.state_dict()[k].copy_(original_state_dict[v])


def cifar10_vggsmall_qfnv2(pretrained=False, **kwargs):
    """VGG small model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGGQFNv2('VGG7', **kwargs)
    if pretrained:
        load_fake_quantized_state_dict(model, torch.load( model_urls['vgg7']) , '../cifar10_vggsmall_qfnv2_map.json')
    return model

def cifar10_vggsmall_q(pretrained=False, **kwargs):
    """VGG small model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGGQ('VGG7', **kwargs)
    if pretrained:
        load_fake_quantized_state_dict(model, torch.load( model_urls['vgg7']) , '../cifar10_vggsmall_qfnv2_map.json')
    return model


class VGGQFNv2(nn.Module):
    def __init__(self, vgg_name, nbits_w=4, nbits_a=4):
        super(VGGQFNv2, self).__init__()
        #self.l2 = l2
        self.features = self._make_layers(cfg[vgg_name], nbits_w=nbits_w, nbits_a=nbits_a)
        scale = 1
        if vgg_name == 'VGG7':
            scale = 16
        self.classifier = nn.Sequential(
            ACT_Q(bit=nbits_a),
            Linear_Q(512 * scale, 10, bit=nbits_w),
            #ACT_Q(bit=nbits_a),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        # out = self.last_actq(out)
        out = self.classifier(out)
        
        return out

    def _make_layers(self, cfg, nbits_w, nbits_a):
        layers = []
        in_channels = 3
        # change to actq+convq by XXX on May 28 2019
        for i, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif in_channels == 3:  # first layer
                layers += [
                    #ACT_fq(bit=-1 if max(nbits_a, nbits_w) <= 0 else nbits_a),
                    ACT_Q(bit=-1 if max(nbits_a, nbits_w) <= 0 else 8, signed=True),
                    Conv2d_quan(in_channels, x, kernel_size=3, padding=1, bias=True,
                              bit=nbits_w
                              ),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True), ]
                in_channels = x
            elif i == 7:  # last layer
                layers += [ACT_Q(bit=nbits_a),
                           Conv2d_quan(in_channels, x, kernel_size=3, padding=1, bias=True,
                                     bit=nbits_w),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True),
                           ]
            else:
                layers += [ACT_Q(bit=nbits_a),
                           Conv2d_quan(in_channels, x, kernel_size=3, padding=1, bias=True,
                                     bit=nbits_w),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True),
                           ]
                in_channels = x
        return nn.Sequential(*layers)

class VGGQ(nn.Module):
    def __init__(self, vgg_name, nbits_w=4, nbits_a=4):
        super(VGGQ, self).__init__()
        #self.l2 = l2
        self.features = self._make_layers(cfg[vgg_name], nbits_w=nbits_w, nbits_a=nbits_a)
        scale = 1
        if vgg_name == 'VGG7':
            scale = 16
        self.classifier = nn.Sequential(
            ACT_Q(bit=nbits_a),
            nn.Linear(512 * scale, 10)
            #Linear_Q(512 * scale, 10, bit=nbits_w),
            #ACT_Q(bit=nbits_a),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        # out = self.last_actq(out)
        out = self.classifier(out)
        
        return out

    def _make_layers(self, cfg, nbits_w, nbits_a):
        layers = []
        in_channels = 3
        # change to actq+convq by XXX on May 28 2019
        for i, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif in_channels == 3:  # first layer
                layers += [
                    #ACT_fq(bit=-1 if max(nbits_a, nbits_w) <= 0 else nbits_a),
                    # ACT_Q(bit=-1 if max(nbits_a, nbits_w) <= 0 else 8, signed=True),
                    #Conv2d_quan(in_channels, x, kernel_size=3, padding=1, bias=True, bit=nbits_w ),
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=True),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True), ]
                in_channels = x
            elif i == 7:  # last layer
                layers += [ACT_Q(bit=nbits_a),
                           Conv2d_quan(in_channels, x, kernel_size=3, padding=1, bias=True,
                                     bit=nbits_w),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True),
                           ]
            else:
                layers += [ACT_Q(bit=nbits_a),
                           Conv2d_quan(in_channels, x, kernel_size=3, padding=1, bias=True,
                                     bit=nbits_w),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True),
                           ]
                in_channels = x
        return nn.Sequential(*layers)


if __name__ == "__main__":
    model = cifar10_vggsmall_qfnv2()
    print(model)
