
from torch  import nn
import torch


class ReferenceFFN(nn.Module):
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.w_gate = nn.Parameter(torch.empty((d_model, d_ffn)))
        self.w_up = nn.Parameter(torch.empty((d_model, d_ffn)))
        self.w_down = nn.Parameter(torch.empty((d_ffn, d_model)))

    def forward(self, x: torch.Tensor, topk_indices: torch.Tensor) -> torch.Tensor:
        G = torch.matmul(x, self.w_gate)
        U = torch.matmul(x, self.w_up)
        sig_G = torch.sigmoid(G)
        silu_G = G * sig_G
        Z = silu_G * U
        mask = torch.zeros_like(Z)
        mask.scatter_(1, topk_indices, 1.0)
        Z_sparse = Z * mask
        out = torch.matmul(Z_sparse, self.w_down)
        return out