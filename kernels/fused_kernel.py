import torch
import triton
import triton.language as tl


@triton.jit
def fused_kernel(
    g_ptr, u_ptr,ind_ptr,
    g_act_ptr,z_act_ptr,u_act_ptr,
    M, dffn,  K,
    stride_gm, stride_gk,
    stride_um, stride_uk,
    stride_zm, stride_zk,
    stride_im, stride_ik,
    BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(axis = 0)
    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < K

    idx_ptr = ind_ptr  + pid_m * stride_im + offs_k*stride_ik
    indeces = tl.load(idx_ptr)

    g_pttrs = g_ptr + pid_m* stride_gm + indeces* stride_gk
    u_pttrs = u_ptr + pid_m* stride_um + indeces* stride_uk

    g_val = tl.load(g_pttrs,mask = mask_k, other = 0)
    u_val = tl.load(u_pttrs,mask = mask_k, other = 0)

    g_val_f32 = g_val.to(tl.float32)
    s_val = (g_val_f32 * tl.sigmoid(g_val_f32)).to(g_val.dtype)
    z_val = s_val * u_val

    z_act_ptr = z_act_ptr + pid_m*stride_zm + offs_k*stride_zk
    g_act_ptr = g_act_ptr + pid_m*stride_zm + offs_k*stride_zk
    u_act_ptr = u_act_ptr + pid_m*stride_zm + offs_k*stride_zk

    tl.store(z_act_ptr, z_val, mask=mask_k)
    tl.store(g_act_ptr, g_val, mask=mask_k)
    tl.store(u_act_ptr, u_val, mask=mask_k)

def fused_sparse_act(G:torch.Tensor,U: torch.Tensor , topk_ind :torch.Tensor):
    M, dffn = G.shape
    _ ,K = topk_ind.shape

    Z_active =  torch.empty(size=(M,K) , device= G.device,dtype = G.dtype)
    G_active = torch.empty(size=(M,K),device=  G.device, dtype = G.dtype)
    U_active = torch.empty(size=(M,K),device= G.device, dtype = G.dtype)

    BLOCK_K = triton.next_power_of_2(K)

    grid = lambda meta: (M,)

    fused_kernel[grid](
        G, U,topk_ind,G_active,Z_active,U_active,
        M=M,dffn= dffn, K=K,
        stride_gm=G.stride(0) , stride_gk = G.stride(1),
        stride_um=U.stride(0) , stride_uk=U.stride(1),
        stride_im=topk_ind.stride(0),stride_ik=topk_ind.stride(1),
        stride_zm=Z_active.stride(0) , stride_zk = Z_active.stride(1),
        BLOCK_K=BLOCK_K
    )

    return Z_active, G_active,U_active


@triton.jit
def fused_backward_kernel(
    grad_z_act_ptr, g_act_ptr, u_act_ptr, indeces_ptr,
    grad_g_dense_ptr, grad_u_dense_ptr,
    M, dffn, K,
    stride_gzm , stride_gzk, stride_gm, stride_gk,
    stride_um,stride_uk, stride_indm, stride_indk,
    stride_ggm , stride_ggk, stride_gum, stride_guk,
    BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    offs_k = tl.arange(0,BLOCK_K)
    mask_k = offs_k < K

    idx_pttrs = indeces_ptr + pid_m*stride_indm + stride_indk*offs_k
    grad_z_act_ptr = grad_z_act_ptr + pid_m*stride_gzm+ stride_gzk*offs_k

    g_act_ptr = g_act_ptr + pid_m*stride_gm+ stride_gk*offs_k
    u_act_ptr = u_act_ptr + pid_m*stride_um + stride_uk*offs_k
    


    indeces = tl.load(idx_pttrs,mask=mask_k,other = 0)
    grad_z = tl.load(grad_z_act_ptr,mask=mask_k,other=0)
    g_val = tl.load(g_act_ptr, mask = mask_k, other =0)
    u_val = tl.load(u_act_ptr,mask=mask_k, other =0)

    sig_g = tl.sigmoid(g_val)
    silu_g = sig_g*g_val

    grad_silu_g = sig_g + silu_g*(1.0-sig_g)
    grad_u_active = grad_z*silu_g
    grad_g_active = grad_z*u_val*grad_silu_g

    grad_g_dense_ptrs = grad_g_dense_ptr + pid_m * stride_ggm + indeces * stride_ggk
    grad_u_dense_ptrs = grad_u_dense_ptr + pid_m * stride_gum + indeces * stride_guk

    tl.store(grad_g_dense_ptrs, grad_g_active, mask=mask_k)
    tl.store(grad_u_dense_ptrs, grad_u_active, mask=mask_k)


class FusedSparseActFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, G,U,topk_ind):
        Z_active,G_active,U_active = fused_sparse_act(G,U,topk_ind)
        ctx.save_for_backward(G_active,U_active,topk_ind)
        ctx.dffn = G.shape[1]
        return Z_active
    @staticmethod

    def backward(ctx, grad_z_active):
        G_active, U_active, topk_ind = ctx.saved_tensors
        M, K = topk_ind.shape
        dffn = ctx.dffn

        grad_g_dense = torch.zeros((M, dffn), device=G_active.device, dtype=G_active.dtype)
        grad_u_dense = torch.zeros((M, dffn), device=G_active.device, dtype=G_active.dtype)

        BLOCK_K = triton.next_power_of_2(K)
        grid = lambda meta: (M,)

        fused_backward_kernel[grid](
            grad_z_active, G_active, U_active, topk_ind,
            grad_g_dense, grad_u_dense,
            M=M, dffn=dffn, K=K,
            stride_gzm=grad_z_active.stride(0), stride_gzk=grad_z_active.stride(1),
            stride_gm=G_active.stride(0), stride_gk=G_active.stride(1),
            stride_um=U_active.stride(0), stride_uk=U_active.stride(1),
            stride_indm=topk_ind.stride(0), stride_indk=topk_ind.stride(1),
            stride_ggm=grad_g_dense.stride(0), stride_ggk=grad_g_dense.stride(1),
            stride_gum=grad_u_dense.stride(0), stride_guk=grad_u_dense.stride(1),
            BLOCK_K=BLOCK_K
        )

        return grad_g_dense, grad_u_dense, None


def apply_fused_sparse_act(G: torch.Tensor, U: torch.Tensor, topk_ind: torch.Tensor) -> torch.Tensor:
    return FusedSparseActFunction.apply(G, U, topk_ind)