import torch
import torch.nn as nn

from ...models.moc_ff_triton import SparseSwiGLUFFN
from ...models.moc_fnn_torch import ReferenceFFN

def test_ffn_equivalence():
    torch.manual_seed(42)
    M, d_model, d_ffn, K = 128, 256, 1024, 32
    
    device = torch.device('cuda')
    dtype = torch.float32
    x_triton = torch.randn((M, d_model), device=device, dtype=dtype, requires_grad=True)
    x_ref = x_triton.clone().detach().requires_grad_(True)
    
    topk_indices = torch.randint(0, d_ffn, (M, K), device=device, dtype=torch.int64)

    triton_ffn =SparseSwiGLUFFN(d_model, d_ffn).to(device=device,dtype=dtype)
    ref_ffn =ReferenceFFN(d_model, d_ffn).to(device=device,dtype=dtype)

    with torch.no_grad():
        ref_ffn.w_gate.copy_(triton_ffn.w_gate)
        ref_ffn.w_up.copy_(triton_ffn.w_up)
        ref_ffn.w_down.copy_(triton_ffn.w_down)
        
    # Прямой проход
    out_triton = triton_ffn(x_triton, topk_indices)
    out_ref = ref_ffn(x_ref, topk_indices)
    
    # Обратный проход
    grad_out = torch.randn_like(out_ref)
    out_triton.backward(grad_out)
    out_ref.backward(grad_out)
    
    rtol, atol = 1e-4, 1e-4
    


    print("Запуск проверки...")
    torch.testing.assert_close(out_triton, out_ref, rtol=rtol, atol=atol)
    print("✓ Прямой проход (Forward): Успешно")
    torch.testing.assert_close(x_triton.grad, x_ref.grad, rtol=rtol, atol=atol)
    print("✓ Градиент по входу (dx): Успешно")
    torch.testing.assert_close(triton_ffn.w_gate.grad, ref_ffn.w_gate.grad, rtol=rtol, atol=atol)
    print("✓ Градиент w_gate: Успешно")
    torch.testing.assert_close(triton_ffn.w_up.grad, ref_ffn.w_up.grad, rtol=rtol, atol=atol)
    print("✓ Градиент w_up: Успешно")
    torch.testing.assert_close(triton_ffn.w_down.grad, ref_ffn.w_down.grad, rtol=rtol, atol=atol)
    print("✓ Градиент w_down: Успешно")

if __name__ == "__main__":
    test_ffn_equivalence()