import torch
import triton 
import triton.language as tl

@triton.jit
def fused_routnig_kernel(
    x_ptr, w_ptr, out_val_ptr, out_idx_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, TOP_K: tl.constexpr
):
    pid = tl.program_id(axis = 0)

    offset_m = pid*BLOCK_M + tl.arange(0, BLOCK_M)
    offset_k = tl.arange(0, BLOCK_K)
    offset_n = tl.arange(0, BLOCK_N)

    x_ptrs = x_ptr +(offset_m[:,None]*stride_xm + offset_k[None,:]*stride_xk)
    w_ptrs = w_ptr + (offset_k[:,None]*stride_wk + offset_n[None,:]*stride_wn)

    accumulator = tl.zeros((BLOCK_M,BLOCK_N),dtype= tl.float32)

    for k in range(tl.cdiv(K,BLOCK_K)):
        x_mask = (offset_m[:,None]<M) &((k* BLOCK_K + offset_k[None,:])<K)
        x_block = tl.load(x_ptrs, mask=x_mask,other =0)
        w_mask = (offset_n[None,:]<N) &((k* BLOCK_K + offset_k[:,None])<K)
        w_block = tl.load(w_ptrs, mask=w_mask,other=0)
        accumulator+=tl.dot(x_block,w_block)
        x_ptrs+=BLOCK_K*stride_xk
        w_ptrs+=BLOCK_K*stride_wk

    channel_ids = tl.arange(0, BLOCK_N)
    channel_ids_broadcast = tl.broadcast_to(channel_ids[None, :], (BLOCK_M, BLOCK_N))
    is_valid_channel = channel_ids_broadcast < N
    accumulator = tl.where(is_valid_channel, accumulator, float('-inf'))
    
    offs_out_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    out_mask = offs_out_m < M
    

    for k_idx in range(TOP_K):
        max_vals = tl.max(accumulator, axis=1)
        max_ids = tl.argmax(accumulator, axis=1)
        

        out_ptrs = out_val_ptr + offs_out_m * TOP_K + k_idx
        idx_ptrs = out_idx_ptr + offs_out_m * TOP_K + k_idx
        

        tl.store(out_ptrs, max_vals, mask=out_mask)
        tl.store(idx_ptrs, max_ids, mask=out_mask)

        is_max = channel_ids_broadcast == max_ids[:, None]
        accumulator = tl.where(is_max, float('-inf'), accumulator)





def fused_routing_forward(x:torch.Tensor, w_gate:torch.Tensor, top_k:int):
    M, K = x.shape
    N, _ = w_gate.shape
    
    x = x.contiguous()
    w_gate = w_gate.contiguous()

    out_vals = torch.empty(size=(M,top_k),device=x.device,dtype = x.dtype)
    out_idx =  torch.empty(size=(M,top_k),device=x.device,dtype = torch.int32)
    #triton autotune
    BLOCK_M = 32
    BLOCK_K = 64

    BLOCK_N = triton.next_power_of_2(N)

    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)

    fused_routnig_kernel[grid](
        x, w_gate, out_vals, out_idx,
        M, N, K,
        x.stride(0), x.stride(1),
        w_gate.stride(0), w_gate.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, TOP_K=top_k
    )

    return out_vals, out_idx

