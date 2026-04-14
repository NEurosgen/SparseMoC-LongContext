import torch
import torch.nn as nn


from models.moc_ffn_triton.SparseSiLUFFN import SparseSiLUFFN 
from models.moc_ffn_torch import ReferenceFFN

def _make_shared_models(M, d_model, d_ffn, K, device, dtype):
    torch.manual_seed(42)
    triton_ffn = SparseSiLUFFN(d_model, d_ffn, top_k=K).to(device=device, dtype=dtype)
    ref_ffn = ReferenceFFN(d_model, d_ffn, top_k=K).to(device=device, dtype=dtype)
    with torch.no_grad():
        ref_ffn.w_gate.copy_(triton_ffn.w_gate)
        ref_ffn.w_up.copy_(triton_ffn.w_up)
        ref_ffn.w_down.copy_(triton_ffn.w_down)
    return triton_ffn, ref_ffn


def test_forward_equivalence():
    M, d_model, d_ffn, K = 128, 256, 1024, 32
    device = torch.device('cuda')
    dtype = torch.float32
    triton_ffn, ref_ffn = _make_shared_models(M, d_model, d_ffn, K, device, dtype)
    torch.manual_seed(123)
    x_triton = torch.randn((M, d_model), device=device, dtype=dtype, requires_grad=True)
    x_ref = x_triton.clone().detach().requires_grad_(True)
    out_triton = triton_ffn(x_triton)
    out_ref = ref_ffn(x_ref)
    rtol, atol = 1e-4, 1e-4
    torch.testing.assert_close(out_triton, out_ref, rtol=rtol, atol=atol)
    print("✓ Forward pass (float32, K=32): OK")


def test_backward_gradients():
    M, d_model, d_ffn, K = 128, 256, 1024, 32
    device = torch.device('cuda')
    dtype = torch.float32
    triton_ffn, ref_ffn = _make_shared_models(M, d_model, d_ffn, K, device, dtype)

    torch.manual_seed(123)
    x_triton = torch.randn((M, d_model), device=device, dtype=dtype, requires_grad=True)
    x_ref = x_triton.clone().detach().requires_grad_(True)
    out_triton = triton_ffn(x_triton)
    out_ref = ref_ffn(x_ref)
    torch.manual_seed(456)
    grad_out = torch.randn_like(out_ref)
    out_triton.backward(grad_out)
    out_ref.backward(grad_out)
    rtol, atol = 1e-4, 1e-4
    torch.testing.assert_close(x_triton.grad, x_ref.grad, rtol=rtol, atol=atol)
    print("✓ Backward: grad_x: OK")
    torch.testing.assert_close(triton_ffn.w_gate.grad, ref_ffn.w_gate.grad, rtol=rtol, atol=atol)
    print("✓ Backward: grad_w_gate: OK")
    torch.testing.assert_close(triton_ffn.w_up.grad, ref_ffn.w_up.grad, rtol=rtol, atol=atol)
    print("✓ Backward: grad_w_up: OK")
    torch.testing.assert_close(triton_ffn.w_down.grad, ref_ffn.w_down.grad, rtol=rtol, atol=atol)
    print("✓ Backward: grad_w_down: OK")


def test_backward_large_k():
    M, d_model, d_ffn, K = 64, 256, 2048, 572
    device = torch.device('cuda')
    dtype = torch.float32
    triton_ffn, ref_ffn = _make_shared_models(M, d_model, d_ffn, K, device, dtype)
    torch.manual_seed(789)
    x_triton = torch.randn((M, d_model), device=device, dtype=dtype, requires_grad=True)
    x_ref = x_triton.clone().detach().requires_grad_(True)
    out_triton = triton_ffn(x_triton)
    out_ref = ref_ffn(x_ref)
    torch.manual_seed(101)
    grad_out = torch.randn_like(out_ref)
    out_triton.backward(grad_out)
    out_ref.backward(grad_out)

    rtol, atol = 1e-3, 1e-3

    torch.testing.assert_close(out_triton, out_ref, rtol=rtol, atol=atol)
    print("✓ Forward (K=572): OK")

    torch.testing.assert_close(x_triton.grad, x_ref.grad, rtol=rtol, atol=atol)
    print("✓ Backward (K=572): grad_x: OK")

    torch.testing.assert_close(triton_ffn.w_gate.grad, ref_ffn.w_gate.grad, rtol=rtol, atol=atol)
    print("✓ Backward (K=572): grad_w_gate: OK")

    torch.testing.assert_close(triton_ffn.w_up.grad, ref_ffn.w_up.grad, rtol=rtol, atol=atol)
    print("✓ Backward (K=572): grad_w_up: OK")

    torch.testing.assert_close(triton_ffn.w_down.grad, ref_ffn.w_down.grad, rtol=rtol, atol=atol)
    print("✓ Backward (K=572): grad_w_down: OK")


def test_backward_float16():
    M, d_model, d_ffn, K = 128, 256, 1024, 64
    device = torch.device('cuda')
    dtype = torch.float16

    triton_ffn, ref_ffn = _make_shared_models(M, d_model, d_ffn, K, device, dtype)

    torch.manual_seed(321)
    x_triton = torch.randn((M, d_model), device=device, dtype=dtype, requires_grad=True)
    x_ref = x_triton.clone().detach().requires_grad_(True)

    out_triton = triton_ffn(x_triton)
    out_ref = ref_ffn(x_ref)

    torch.manual_seed(654)
    grad_out = torch.randn_like(out_ref)
    out_triton.backward(grad_out)
    out_ref.backward(grad_out)

    rtol, atol = 5e-2, 5e-2

    torch.testing.assert_close(out_triton, out_ref, rtol=rtol, atol=atol)
    print("✓ Forward (float16): OK")

    torch.testing.assert_close(x_triton.grad, x_ref.grad, rtol=rtol, atol=atol)
    print("✓ Backward (float16): grad_x: OK")

    torch.testing.assert_close(triton_ffn.w_gate.grad, ref_ffn.w_gate.grad, rtol=rtol, atol=atol)
    print("✓ Backward (float16): grad_w_gate: OK")

    torch.testing.assert_close(triton_ffn.w_up.grad, ref_ffn.w_up.grad, rtol=rtol, atol=atol)
    print("✓ Backward (float16): grad_w_up: OK")

    torch.testing.assert_close(triton_ffn.w_down.grad, ref_ffn.w_down.grad, rtol=rtol, atol=atol)
    print("✓ Backward (float16): grad_w_down: OK")


def test_backward_3d_input():
    B, S, d_model, d_ffn, K = 4, 64, 256, 1024, 32
    device = torch.device('cuda')
    dtype = torch.float32
    triton_ffn, ref_ffn = _make_shared_models(B * S, d_model, d_ffn, K, device, dtype)
    torch.manual_seed(999)
    x_triton = torch.randn((B, S, d_model), device=device, dtype=dtype, requires_grad=True)
    x_ref = x_triton.clone().detach().requires_grad_(True)
    out_triton = triton_ffn(x_triton)
    out_ref = ref_ffn(x_ref)

    torch.manual_seed(888)
    grad_out = torch.randn_like(out_ref)
    out_triton.backward(grad_out)
    out_ref.backward(grad_out)

    rtol, atol = 1e-4, 1e-4

    torch.testing.assert_close(out_triton, out_ref, rtol=rtol, atol=atol)
    print("✓ Forward (3D input): OK")

    torch.testing.assert_close(x_triton.grad, x_ref.grad, rtol=rtol, atol=atol)
    print("✓ Backward (3D input): grad_x: OK")

    torch.testing.assert_close(triton_ffn.w_gate.grad, ref_ffn.w_gate.grad, rtol=rtol, atol=atol)
    print("✓ Backward (3D input): grad_w_gate: OK")

    torch.testing.assert_close(triton_ffn.w_up.grad, ref_ffn.w_up.grad, rtol=rtol, atol=atol)
    print("✓ Backward (3D input): grad_w_up: OK")

    torch.testing.assert_close(triton_ffn.w_down.grad, ref_ffn.w_down.grad, rtol=rtol, atol=atol)
    print("✓ Backward (3D input): grad_w_down: OK")


if __name__ == "__main__":
    print("=" * 60)
    print("Тест 1: Forward pass (float32, K=32)")
    print("=" * 60)
    test_forward_equivalence()

    print("\n" + "=" * 60)
    print("Тест 2: Backward pass — все градиенты (float32, K=32)")
    print("=" * 60)
    test_backward_gradients()

    print("\n" + "=" * 60)
    print("Тест 3: Backward pass — большой K=572 (shared memory)")
    print("=" * 60)
    test_backward_large_k()

    print("\n" + "=" * 60)
    print("Тест 4: Backward pass — float16")
    print("=" * 60)
    test_backward_float16()

    print("\n" + "=" * 60)
    print("Тест 5: Backward pass — 3D вход (batch, seq, dim)")
    print("=" * 60)
    test_backward_3d_input()

    print("\n" + "=" * 60)
    print("Все тесты пройдены! ✓")
    print("=" * 60)