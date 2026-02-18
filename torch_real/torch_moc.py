import torch 
from torch import nn
from moc_lib.kernels.fuesd_routnig import fused_routing_forward

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
        mask = torch.zeros_like(gOut).scatter_(-1, topk_indeces, 1.0)

        out = self.fwDown(hid * mask * activate)
        return out
    
class MocFFn_kenrel(nn.Module):
    def __init__(self, input, hidden, top_k ):
        super().__init__()
        self.fwUp = nn.Linear(in_features=input,out_features=hidden, bias=False)
        self.gate = nn.Linear(in_features=input,out_features=hidden, bias=False)
        self.fwDown = nn.Linear(in_features=hidden, out_features=input, bias = False)
        self.top_k = top_k
        self.silu= nn.SiLU()

    def forward(self , x):
        orig_shape = x.shape
        x_2d = x.view(-1, orig_shape[-1])
        
        topk_vals, topk_idx = fused_routing_forward(x_2d, self.gate.weight, self.top_k)
        
        activate_sparse = self.silu(topk_vals)
        
        hid = self.fwUp(x)
        
        gOut_dense = torch.zeros_like(hid).scatter_(-1, topk_idx.long(), activate_sparse)
        out = self.fwDown(hid * gOut_dense)
        return out
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch_md = MocFFn(3, 3, 2).to(device)
triton_md = MocFFn_kenrel(3, 3, 2).to(device)


with torch.no_grad():
    triton_md.fwUp.weight.copy_(torch_md.fwUp.weight)
    triton_md.gate.weight.copy_(torch_md.gate.weight)
    triton_md.fwDown.weight.copy_(torch_md.fwDown.weight)

x = torch.randn(size=(20, 3), device=device)

out_torch = torch_md(x)
out_triton = triton_md(x)

# Проверяем на математическую эквивалентность
torch.testing.assert_close(out_torch, out_triton, rtol=1e-4, atol=1e-4)