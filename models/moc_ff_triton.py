from kernels.fused_kernel import apply_fused_sparse_act
from kernels.gather_matmul import apply_sparse_to_dense_linear
import torch
import torch.nn as nn
import numpy as np

class SparseSiLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, top_k = 256):
        super().__init__()
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.top_k = top_k
        self.w_gate = nn.Parameter(torch.empty((d_model, d_ffn)))
        self.w_up = nn.Parameter(torch.empty((d_model, d_ffn)))
        self.w_down = nn.Parameter(torch.empty((d_ffn, d_model)))
        
        self.reset_parameters()

    def reset_parameters(self):

        nn.init.kaiming_uniform_(self.w_gate, a=np.sqrt(5.))
        nn.init.kaiming_uniform_(self.w_up, a=np.sqrt(5.))
        nn.init.kaiming_uniform_(self.w_down, a=np.sqrt(5.))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_flat = x.reshape(-1, self.d_model)

        G = torch.matmul(x_flat, self.w_gate)
        U = torch.matmul(x_flat, self.w_up)
        _, topk_indices = torch.topk(G, self.top_k, dim=-1)
        Z_active = apply_fused_sparse_act(G, U, topk_indices)
        
        out_flat = apply_sparse_to_dense_linear(Z_active, topk_indices, self.w_down)
        out = out_flat.reshape(*original_shape[:-1], self.d_model)
        return out