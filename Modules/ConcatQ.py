
import torch
import torch.nn as nn
from models.modules import ActQ
from models.modules.quantize import ln_error
import ipdb

__all__ = ['Concat', 'ConcatQ']


class Concat(nn.Module):
    def __init__(self, dim=0):
        super(Concat, self).__init__()
        self.dim = dim

    def forward(self, *seq):
        return torch.cat(seq, dim=self.dim)


class ConcatQ(nn.Module):
    def __init__(self, dim=0, nbits=4, signed=False, l2=True):
        super(ConcatQ, self).__init__()
        self.dim = dim
        self.actq = ActQ(nbits=nbits, signed=signed, l2=l2)

    def forward(self, *seq):
        concat_q = self.actq(torch.cat(seq, dim=self.dim))
        seq_q = []
        for input_ in seq:
            if self.actq.running_scale is not None:
                scale = self.actq.running_scale.detach()
                _, x_clip, y = ln_error(input_, self.nbits, scale, is_act=True, l2=self.l2)
                out_q = y.detach() + x_clip - x_clip.detach()
            else:
                out_q = input_
            seq_q.append(out_q)
        return concat_q, seq_q
