
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



class RoundFn_act(Function):
    @staticmethod
    def forward(ctx, input, alpha, pwr_coef, bit, signed):
        
        if signed == True:
            x_alpha_div = (input  / alpha ).round().clamp( min =-(pwr_coef), max = (pwr_coef-1)) *  alpha
        else:
            x_alpha_div = (input  / alpha ).round().clamp( min =0, max = (pwr_coef-1)) *  alpha
        ctx.pwr_coef = pwr_coef
        ctx.bit      = bit
        ctx.signed   = signed
        ctx.save_for_backward(input, alpha)
        return x_alpha_div 
    @staticmethod
    def backward(ctx, grad_output):
        input, alpha = ctx.saved_tensors
        pwr_coef = ctx.pwr_coef
        bit = ctx.bit
        signed = ctx.signed
        if signed == True:
            low_bound = -(pwr_coef)
        else:
            low_bound = 0
        quan_Em =  (input  / alpha   ).round().clamp( min =low_bound, max = (pwr_coef-1)) * alpha 
        quan_El =  (input / ( alpha  / 2)   ).round().clamp( min =low_bound, max = (pwr_coef-1)) * ( alpha  / 2)
        quan_Er = (input / ( alpha * 2)  ).round().clamp( min =low_bound, max = (pwr_coef-1)) * ( alpha * 2)
        El = torch.sum(torch.pow((input - quan_El), 2 ))
        Er = torch.sum(torch.pow((input - quan_Er), 2 ))
        Em = torch.sum(torch.pow((input - quan_Em), 2 ))
        d_better = torch.Tensor([El, Em, Er]).argmin() -1
        delta_G = (-1) * (torch.pow(alpha , 2)) * (  d_better) 

        grad_input = grad_output.clone()
        if signed == True:
            grad_input = torch.where((input) < ( (-1) * pwr_coef  * alpha ) , torch.full_like(grad_input,0), grad_input ) 
            grad_input = torch.where((input) > ((pwr_coef   - 1) * alpha ),  torch.full_like(grad_input,0), grad_input)    
        else:
            grad_input = torch.where( (input) < 0 , torch.full_like(grad_input,0), grad_input )
            grad_input = torch.where((input) > ((pwr_coef * 2 - 1) * alpha ),  torch.full_like(grad_input,0), grad_input)
            

        return  grad_input, delta_G, None, None, None


class ACT_Q(nn.Module):
    def __init__(self,  bit=32 , signed = False):
        super(ACT_Q, self).__init__()
        self.bit        = bit
        self.signed = signed
        self.pwr_coef   = 2** (bit - 1)
        self.alpha = Parameter(torch.rand(1)).cuda()    
        self.round_fn = RoundFn_act.apply
        self.alpha_qfn = quan_fn_alpha()
    def forward(self, input):
        if( self.bit == 32 ):
            return input
        else:
            alpha = self.alpha_qfn(self.alpha)
            act = self.round_fn( input, alpha, self.pwr_coef, self.bit, self.signed)
            return act

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