import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function, Variable
from torch.nn.parameter import Parameter
from torch.distributions import Bernoulli

torch.set_default_tensor_type('torch.cuda.FloatTensor')

#from .Quan_Act import RoundFn_act as RoundFn_LLSQ
from .Conv2d_quan import RoundFn_LLSQ, RoundFn_Bias, quan_alpha

class Linear_Q(nn.Linear):
    def __init__(self, in_features, out_features,  bias=True, bit=32, extern_init=False, init_model=nn.Sequential()):
        super(Linear_Q, self).__init__(
            in_features, out_features,  bias)
        self.bit = bit
        self.pwr_coef = 2** (bit - 1)
        
        self.alpha_w = Parameter(torch.rand( 1)).cuda()
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        if extern_init:
            param=list(init_model.parameters())
            self.weight=Parameter(param[0])
            if bias:
                self.bias=Parameter(param[1])
        self.Round_w = RoundFn_LLSQ.apply
        self.Round_b = RoundFn_Bias.apply

    def forward(self, x):
        if self.bit == 32:
            return F.linear(
                x, self.weight, self.bias)
        else:
            assert not torch.isnan(x).any(), "Linear Input should not be 'nan'"
            wq = self.Round_w(self.weight, self.alpha_w, self.pwr_coef, self.bit)
            b  = self.Round_b(self.bias  , self.alpha_w, self.pwr_coef, self.bit)
            return F.linear(
                x, wq, b)