import torch
import torch.nn as nn
from models.moc_ff_triton import SparseSwiGLUFFN
from models.dense_fnn import DenseFFn
from models.moc_fnn_torch import ReferenceFFN






class SingleLayerTransformer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_module: nn.Module):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = ffn_module

    def forward(self, x: torch.Tensor, topk_indices: torch.Tensor = None) -> torch.Tensor:
        residual = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = residual + attn_out
        residual = x
        x_norm = self.norm2(x)
        
        if topk_indices is not None:
            B, S, D = x_norm.shape
            x_flat = x_norm.view(-1, D)
            indices_flat = topk_indices.view(-1, topk_indices.shape[-1])
            
            ffn_out = self.ffn(x_flat, indices_flat)
            ffn_out = ffn_out.view(B, S, D)
        else:
            ffn_out = self.ffn(x_norm)
            
        return residual + ffn_out
    

def run_latency_benchmark():
    device = torch.device('cuda')
    dtype = torch.float32
    
    B, S = 8, 512    
    d_model = 1024        
    n_heads = 4         
    d_ffn = 2048       
    K = 128               

    print(f"Измерение latency (МС) | B={B}, S={S}, d_model={d_model}, d_ffn={d_ffn}, K={K}")
    print("-" * 65)

    def measure_time(model_name: str, model: nn.Module, is_sparse: bool, num_warmup=10, num_iters=50):
        model = model.to(device=device, dtype=dtype)
        model.train()

        x = torch.randn((B, S, d_model), device=device, dtype=dtype, requires_grad=True)
        topk_indices = torch.randint(0, d_ffn, (B, S, K), device=device, dtype=torch.int64) if is_sparse else None

        for _ in range(num_warmup):
            out = model(x, topk_indices)
            loss = out.sum()
            loss.backward()
            x.grad = None
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = None

        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(num_iters):
            out = model(x, topk_indices)
            loss = out.sum()
            loss.backward()
            
            x.grad = None
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = None
                    
        end_event.record()
        torch.cuda.synchronize()
        total_time_ms = start_event.elapsed_time(end_event)
        avg_time_ms = total_time_ms / num_iters
        
        print(f"{model_name:<25}: {avg_time_ms:.3f} ms / итерация")
        
        del model, x, out, loss
        if is_sparse:
            del topk_indices
        torch.cuda.empty_cache()

    measure_time(
        "Dense FFN (Baseline)", 
        SingleLayerTransformer(d_model, n_heads, DenseFFn(d_model, d_ffn)), 
        is_sparse=False
    )

    measure_time(
        "PyTorch Sparse (Gather)", 
        SingleLayerTransformer(d_model, n_heads, ReferenceFFN(d_model, d_ffn)), 
        is_sparse=True
    )

    measure_time(
        "Triton Sparse (Fused)", 
        SingleLayerTransformer(d_model, n_heads, SparseSwiGLUFFN(d_model, d_ffn)), 
        is_sparse=True
    )

if __name__ == "__main__":
    run_latency_benchmark()