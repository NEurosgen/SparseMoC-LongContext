
from torch  import nn
import torch
import numpy as np

class ReferenceFFN(nn.Module):
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.w_gate = nn.Parameter(torch.empty((d_model, d_ffn)))
        self.w_up = nn.Parameter(torch.empty((d_model, d_ffn)))
        self.w_down = nn.Parameter(torch.empty((d_ffn, d_model)))
        
        nn.init.kaiming_uniform_(self.w_gate, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.w_up, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.w_down, a=np.sqrt(5))

    def forward(self, x: torch.Tensor, topk_indices: torch.Tensor) -> torch.Tensor:
        M, d_model = x.shape
        d_ffn = self.w_down.shape[0]
        G = torch.matmul(x, self.w_gate)
        U = torch.matmul(x, self.w_up)

        G_active = torch.gather(G, 1, topk_indices)
        U_active = torch.gather(U, 1, topk_indices)

        sig_G = torch.sigmoid(G_active)
        Z_active = (G_active * sig_G) * U_active
        
        Z_dense = torch.zeros((M, d_ffn), device=x.device, dtype=x.dtype)
        Z_dense.scatter_(1, topk_indices, Z_active)
    
        out = torch.matmul(Z_dense, self.w_down)
        
        return out