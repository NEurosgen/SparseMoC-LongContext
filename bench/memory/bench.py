import torch
from torch import nn
import gc
from ...models.moc_ff_triton import SparseSwiGLUFFN
from ...models.dense_fnn import DenseFFn
from ...models.moc_fnn_torch import ReferenceFFN





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
    

    
def run_memory_benchmark():

    device = torch.device('cuda')
    dtype = torch.float32
    
    B, S = 8, 512      
    M = B * S            
    d_model = 1024       
    n_heads = 2          
    d_ffn = 2048          
    K = 128               

    print(f"Параметры: Токенов={M}, d_model={d_model}, d_ffn={d_ffn}, K={K}")
    print("-" * 50)

    def measure_peak_vram(model_name: str, model: nn.Module, is_sparse: bool):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model = model.to(device=device, dtype=dtype)
        model.train() 

        x = torch.randn((B, S, d_model), device=device, dtype=dtype, requires_grad=True)
        topk_indices = torch.randint(0, d_ffn, (B, S, K), device=device, dtype=torch.int64) if is_sparse else None
        out = model(x, topk_indices)
        

        loss = out.sum()
        loss.backward()

        peak_memory_bytes = torch.cuda.max_memory_allocated()
        peak_memory_mb = peak_memory_bytes / (1024 ** 2)
        
        print(f"{model_name:>20}: {peak_memory_mb:.2f} MB")
        
        del model, x, out, loss
        if is_sparse:
            del topk_indices


    measure_peak_vram(
        "Dense FFN (Baseline)", 
        SingleLayerTransformer(d_model, n_heads, DenseFFn(d_model, d_ffn)), 
        is_sparse=False
    )

    # 2. Замер: Разреженный PyTorch FFN
    measure_peak_vram(
        "PyTorch Sparse FFN", 
        SingleLayerTransformer(d_model, n_heads, ReferenceFFN(d_model, d_ffn)), 
        is_sparse=True
    )

    # 3. Замер: Разреженный Triton FFN
    measure_peak_vram(
        "Triton Sparse FFN", 
        SingleLayerTransformer(d_model, n_heads, SparseSwiGLUFFN(d_model, d_ffn)), 
        is_sparse=True
    )

if __name__ == "__main__":
    run_memory_benchmark()