from torch  import nn
import torch
class DenseFFn(nn.Module):
    def __init__(self, d_model, dffn):
        super().__init__()
        self.w_gate = nn.Linear(d_model,dffn,bias = False)
        self.w_up = nn.Linear(d_model,dffn,bias = False)
        self.w_down = nn.Linear(dffn, d_model,bias = False)
    def forward(self, x):
        G = self.w_gate(x)
        U = self.w_up(x)
        Z = (G * torch.sigmoid(G)) * U
        return self.w_down(Z)