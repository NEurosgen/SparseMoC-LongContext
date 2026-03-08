from kernels.fused_kernel import apply_fused_sparse_act
from kernels.gather_matmul import apply_sparse_to_dense_linear
import torch
import torch.nn as nn
import numpy as np

class SparseSwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.d_model = d_model
        self.d_ffn = d_ffn

        self.w_gate = nn.Parameter(torch.empty((d_model, d_ffn)))
        self.w_up = nn.Parameter(torch.empty((d_model, d_ffn)))
        self.w_down = nn.Parameter(torch.empty((d_ffn, d_model)))
        
        self.reset_parameters()

    def reset_parameters(self):

        nn.init.kaiming_uniform_(self.w_gate, a=np.sqrt(5.))
        nn.init.kaiming_uniform_(self.w_up, a=np.sqrt(5.))
        nn.init.kaiming_uniform_(self.w_down, a=np.sqrt(5.))

    def forward(self, x: torch.Tensor, topk_indices: torch.Tensor) -> torch.Tensor:

        G = torch.matmul(x, self.w_gate)
        U = torch.matmul(x, self.w_up)
        
        Z_active = apply_fused_sparse_act(G, U, topk_indices)
        
        out = apply_sparse_to_dense_linear(Z_active, topk_indices, self.w_down)
        
        return out