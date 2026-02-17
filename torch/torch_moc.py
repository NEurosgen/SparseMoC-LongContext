import torch 
from torch import nn


class FFn(nn.Module):
    def __init__(self, input, hidden):
        super().__init__()
        self.fwUp = nn.Linear(in_features=input,out_features=hidden, bias=False)
        self.gate = nn.Linear(in_features=input,out_features=hidden, bias=False)
        self.fwDown = nn.Linear(in_features=hidden, out_features=input, bias = False)
        self.silu= nn.SiLU()
    def forward(self , x):
        hid = self.fwUp(x)
        gOut = self.silu(self.gate(x))
        out = self.fwDown(hid*gOut)
        return out


class MocFFn(nn.Module):
    def __init__(self, input, hidden, top_k ):
        super().__init__()
        self.fwUp = nn.Linear(in_features=input,out_features=hidden, bias=False)
        self.gate = nn.Linear(in_features=input,out_features=hidden, bias=False)
        self.fwDown = nn.Linear(in_features=hidden, out_features=input, bias = False)
        self.top_k = top_k
        self.silu= nn.SiLU()
    def forward(self , x):
        hid = self.fwUp(x)
        gOut = self.gate(x)
        activate = self.silu(gOut)
        _, topk_indeces = torch.topk(gOut, self.top_k, dim=-1)
        mask = torch.zeros_like(gOut).scatter_(-1, topk_indeces,1)

        out = self.fwDown(hid*mask*activate)
        return out
