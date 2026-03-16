import torch
import triton
import triton.language as tl


@triton.jit
def unpack_z_active(z_active_ptr, ind_ptr, z_dense_ptr,
                    M, K, 
                    stride_zm, stride_zk,
                    stride_im, stride_ik,
                    stride_dm, stride_dd,
                    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)

    off_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = off_m < M

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        mask_mk = mask_m[:, None] & mask_k[None, :]

        z_ptr = z_active_ptr + off_m[:, None] * stride_zm + offs_k[None, :] * stride_zk
        curr_ind_ptr = ind_ptr + off_m[:, None] * stride_im + offs_k[None, :] * stride_ik

        z_act = tl.load(z_ptr, mask=mask_mk, other=0.0)
        indexes = tl.load(curr_ind_ptr, mask=mask_mk, other=0)
        dense_ptr = z_dense_ptr + off_m[:, None] * stride_dm + indexes * stride_dd
        tl.store(dense_ptr, z_act, mask=mask_mk)

def sparse_to_dense(z_active, topk_indeces, d_ffn):
    M, K = z_active.shape
    z_dense = torch.zeros(size=(M, d_ffn), device=z_active.device, dtype=z_active.dtype)

    BLOCK_M = 64
    BLOCK_K = min(128, triton.next_power_of_2(K))
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), )

    unpack_z_active[grid](
        z_active, topk_indeces, z_dense,
        M, K,
        z_active.stride(0), z_active.stride(1),
        topk_indeces.stride(0), topk_indeces.stride(1),
        z_dense.stride(0), z_dense.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K
    )
    
    return z_dense


@triton.jit
def pack_grad_z_kernel(
    grad_z_dense_ptr, indices_ptr, grad_z_active_ptr,
    M, K, d_ffn,
    stride_gdm, stride_gdd,
    stride_im, stride_ik,
    stride_gam, stride_gak,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        mask_mk = mask_m[:, None] & mask_k[None, :]

        idx_ptrs = indices_ptr + offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik
        indices = tl.load(idx_ptrs, mask=mask_mk, other=0)

        grad_dense_ptrs = grad_z_dense_ptr + offs_m[:, None] * stride_gdm + indices * stride_gdd
        grad_vals = tl.load(grad_dense_ptrs, mask=mask_mk, other=0.0)

        grad_active_ptrs = grad_z_active_ptr + offs_m[:, None] * stride_gam + offs_k[None, :] * stride_gak
        tl.store(grad_active_ptrs, grad_vals, mask=mask_mk)

def backward_sparse_indexed_z(grad_D: torch.Tensor, W_down: torch.Tensor, topk_indices: torch.Tensor, Z_dense_forward: torch.Tensor = None):

    M, K = topk_indices.shape
    d_ffn = W_down.shape[0]
    grad_Z_dense = torch.matmul(grad_D, W_down.t())
    grad_Z_active = torch.empty((M, K), device=grad_D.device, dtype=grad_D.dtype)
    
    BLOCK_M = 64
    BLOCK_K = min(128, triton.next_power_of_2(K))
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)

    pack_grad_z_kernel[grid](
        grad_Z_dense, topk_indices, grad_Z_active,
        M, K, d_ffn,
        grad_Z_dense.stride(0), grad_Z_dense.stride(1),
        topk_indices.stride(0), topk_indices.stride(1),
        grad_Z_active.stride(0), grad_Z_active.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K
    )
    
    grad_W_down = None
    if Z_dense_forward is not None:
        grad_W_down = torch.matmul(Z_dense_forward.t(), grad_D)
        
    return grad_Z_active, grad_W_down



class FusedSparseToDenseLinear(torch.autograd.Function):

    @staticmethod
    def forward(ctx, z_active: torch.Tensor, topk_indices: torch.Tensor, W_down: torch.Tensor):
        d_ffn = W_down.shape[0]
        z_dense = sparse_to_dense(z_active, topk_indices, d_ffn)
        out = torch.matmul(z_dense, W_down)
        ctx.save_for_backward(W_down, topk_indices, z_active)
        ctx.d_ffn = d_ffn
        
        return out

    @staticmethod
    def backward(ctx, grad_out):

        W_down, topk_indices, z_active = ctx.saved_tensors
        # Пересоздаём z_dense из z_active + topk_indices (дешёвая операция)
        z_dense = sparse_to_dense(z_active, topk_indices, ctx.d_ffn)
        grad_z_active, grad_W_down = backward_sparse_indexed_z(
            grad_D=grad_out, 
            W_down=W_down, 
            topk_indices=topk_indices, 
            Z_dense_forward=z_dense
        )
        

        return grad_z_active, None, grad_W_down


def apply_sparse_to_dense_linear(z_active: torch.Tensor, topk_indices: torch.Tensor, W_down: torch.Tensor) -> torch.Tensor:
    return FusedSparseToDenseLinear.apply(z_active, topk_indices, W_down)
