# %%
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function, Variable
from torch.nn import Parameter
from torch.distributions import Bernoulli

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt




__all__ = [
    'QuantConv2d', 'RoundFn_LLSQ', 'RoundFn_Bias', 'quan_alpha'
]


class RoundFn_LLSQ(Function):
    @staticmethod
    def forward(ctx, input, alpha, pwr_coef, bit):
      
        alpha = quan_alpha(alpha, 16)
        x_alpha_div = (input  / alpha  ).round().clamp( min =-(pwr_coef), max = (pwr_coef-1)) * alpha  
        
        ctx.pwr_coef = pwr_coef
        ctx.bit      = bit
        ctx.save_for_backward(input, alpha)
        return x_alpha_div 
    @staticmethod
    def backward(ctx, grad_output):
        input, alpha = ctx.saved_tensors
        pwr_coef = ctx.pwr_coef
        bit      = ctx.bit
        quan_Em =  (input  / (alpha ) ).round().clamp( min =-(pwr_coef), max = (pwr_coef-1)) * alpha  
        
       
        quan_El =  (input / ((alpha ) / 2) ).round().clamp( min =-(pwr_coef), max = (pwr_coef-1)) * (alpha  / 2) 
        
        
        quan_Er = (input / ((alpha ) * 2) ).round().clamp( min =-(pwr_coef), max = (pwr_coef-1)) * (alpha  * 2) 
        
        if len(list(input.size())) > 3:
            El = torch.sum(torch.pow((input - quan_El), 2 ), dim = (1,2,3))
            Er = torch.sum(torch.pow((input - quan_Er), 2 ), dim = (1,2,3))
            Em = torch.sum(torch.pow((input - quan_Em), 2 ), dim = (1,2,3))
            d_better = torch.argmin( torch.stack([El, Em, Er], dim=0), dim=0) -1
            delta_G = - (torch.pow(alpha , 2)) * ( d_better.view(list(alpha.size()))) 
        else:
            El = torch.sum(torch.pow((input - quan_El), 2 ))
            Er = torch.sum(torch.pow((input - quan_Er), 2 ))
            Em = torch.sum(torch.pow((input - quan_Em), 2 ))
            d_better = torch.Tensor([El, Em, Er]).argmin() -1
            delta_G = (-1) * (torch.pow(alpha , 2)) * ( d_better) 
            


        grad_input = grad_output.clone()

        
        return  grad_input, delta_G, None, None


class RoundFn_Bias(Function):
    @staticmethod
    def forward(ctx, input, alpha, pwr_coef, bit):
        alpha = quan_alpha(alpha, 16)
        alpha = torch.reshape(alpha, (-1,))
        x_alpha_div = (input  / alpha).round().clamp( min =-(pwr_coef), max = (pwr_coef-1)) * alpha 
        ctx.pwr_coef = pwr_coef
        
        return x_alpha_div 
    @staticmethod
    def backward(ctx, grad_output):
        
        
        grad_input = grad_output.clone()
        return  grad_input, None, None, None


class QuantConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, bit=32, extern_init=False, init_model=nn.Sequential()):
        super(QuantConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias)
        self.bit = bit
        self.pwr_coef =  2**(bit - 1) 
        self.Round_w = RoundFn_LLSQ.apply
        self.Round_b = RoundFn_Bias.apply
        self.bias_flag = bias
        self.alpha_w = Parameter(torch.rand( out_channels,1,1,1)).cuda()
        self.alpha_qfn = quan_fn_alpha()
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.alpha_w, mode='fan_out', nonlinearity='relu')
        if extern_init:
            param=list(init_model.parameters())
            self.weight=Parameter(param[0])
            if bias:
                self.bias=Parameter(param[1])
    def forward(self, x):
        if self.bit == 32:
            return F.conv2d(
                x, self.weight, self.bias, self.stride, self.padding,
                self.dilation, self.groups)
        else:

            assert not torch.isnan(x).any(), "Conv2d Input should not be 'nan'"
            alpha_w = self.alpha_qfn(self.alpha_w)
            if torch.isnan(self.alpha_w).any() or torch.isinf(self.alpha_w).any():
                #print(self.alpha_w)
                assert not torch.isnan(wq).any(), self.alpha_w
                assert not torch.isinf(wq).any(), self.alpha_w

            wq =  self.Round_w(self.weight, alpha_w, self.pwr_coef, self.bit)
            if self.bias_flag == True:
                LLSQ_b  = self.Round_b(self.bias, alpha_w, self.pwr_coef, self.bit)
            else:
                LLSQ_b = self.bias
            
            assert not torch.isnan(self.weight).any(), "Weight should not be 'nan'"
            if torch.isnan(wq).any() or torch.isinf(wq).any():
                print(self.alpha_w)
                assert not torch.isnan(wq).any(), "Conv2d Weights should not be 'nan'"
                assert not torch.isinf(wq).any(), "Conv2d Weights should not be 'nan'"
            
            return F.conv2d(
                x,  wq, LLSQ_b, self.stride, self.padding, self.dilation,
                self.groups)

def quan_alpha(alpha, bits):
    if(bits==32):
        alpha_q = alpha
    else:
        q_code  = bits - torch.ceil( torch.log2( torch.max(alpha)) + 1 - 1e-5 )
        alpha_q = torch.clamp( torch.round( alpha * (2**q_code)), -2**(bits - 1), 2**(bits - 1) - 1 ) / (2**q_code)
    return alpha_q

class quan_fn_alpha(nn.Module):
    def __init__(self,  bit=32 ):
        super(quan_fn_alpha, self).__init__()
        self.bits = bit
        self.pwr_coef   = 2** (bit - 1)
    def forward(self, alpha):
        q_code  = self.bits - torch.ceil( torch.log2( torch.max(alpha)) + 1 - 1e-5 )
        alpha_q = torch.clamp( torch.round( alpha * (2**q_code)), -2**(self.bits - 1), 2**(self.bits - 1) - 1 ) / (2**q_code)
        return alpha_q
    def backward(self, input):
        
        return input