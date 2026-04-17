import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.moc_ffn_triton.SparseSiLUFFN import SparseSiLUFFN

def free_gpu():
    """Очистка GPU кэша и сброс статистики peak memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def free_model(model):
    """Полностью выгружает модель из GPU/RAM."""
    del model
    free_gpu()

def load_model(model_path, sparse_weights_path=None, device='cuda',
               attn_implementation='flash_attention_2'):
    """
    Загрузка Qwen3-0.6B с опциональной заменой FFN на SparseSiLUFFN.

    Args:
        model_path: путь к HuggingFace модели
        sparse_weights_path: путь к sparse_ffn_weights.pt (None = dense модель)
        device: 'cuda' или 'cpu'
    """
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype="auto", device_map=device,
            attn_implementation=attn_implementation,
        )
    except Exception as e:
        if attn_implementation == 'flash_attention_2':
            print(f"flash_attention_2 не доступен ({e}), fallback → sdpa")
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype="auto", device_map=device,
                attn_implementation="sdpa",
            )
        else:
            raise

    if sparse_weights_path is None:
        return model

    sparse_weights = torch.load(sparse_weights_path, map_location=device, weights_only=True)
    model_dtype = model.dtype

    for key, layer_weights in sparse_weights.items():
        layer_idx = int(key.split("_")[1])

        new_mlp = SparseSiLUFFN(
            d_model=layer_weights["d_model"],
            d_ffn=layer_weights["d_ffn"],
            top_k=layer_weights["top_k"],
        )
        new_mlp = new_mlp.to(device=device, dtype=model_dtype)

        with torch.no_grad():
            new_mlp.w_gate.copy_(layer_weights["w_gate"].to(model_dtype))
            new_mlp.w_up.copy_(layer_weights["w_up"].to(model_dtype))
            new_mlp.w_down.copy_(layer_weights["w_down"].to(model_dtype))

        model.model.layers[layer_idx].mlp = new_mlp

    return model



def detailed_memory_stats(model=None, optimizer=None, label=""):
    """
    Собирает детальную статистику GPU памяти.

    Returns:
        dict с полями:
            - allocated_mb: текущее выделение
            - reserved_mb: зарезервировано CUDA allocator
            - peak_allocated_mb: пиковое выделение с момента последнего reset
            - model_params_mb: память под параметры модели
            - gradients_mb: память под градиенты
            - optimizer_states_mb: приблизительная память optimizer
    """
    if not torch.cuda.is_available():
        return {"label": label, "status": "no_cuda"}

    stats = {
        "label": label,
        "allocated_mb": round(torch.cuda.memory_allocated() / (1024 ** 2), 2),
        "reserved_mb": round(torch.cuda.memory_reserved() / (1024 ** 2), 2),
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2),
    }

    if model is not None:
        param_bytes = sum(
            p.numel() * p.element_size() for p in model.parameters()
        )
        grad_bytes = sum(
            p.grad.numel() * p.grad.element_size()
            for p in model.parameters() if p.grad is not None
        )
        stats["model_params_mb"] = round(param_bytes / (1024 ** 2), 2)
        stats["gradients_mb"] = round(grad_bytes / (1024 ** 2), 2)

    if optimizer is not None:
        opt_bytes = 0
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state.get(p, {})
                for v in state.values():
                    if isinstance(v, torch.Tensor):
                        opt_bytes += v.numel() * v.element_size()
        stats["optimizer_states_mb"] = round(opt_bytes / (1024 ** 2), 2)


    known = stats.get("model_params_mb", 0) + stats.get("gradients_mb", 0) + stats.get("optimizer_states_mb", 0)
    stats["activations_and_other_mb"] = round(stats["allocated_mb"] - known, 2)

    return stats


def print_memory_stats(stats):
    """Красивый вывод статистики памяти."""
    label = stats.get("label", "")
    print(f"\n  ┌── Memory: {label} {'─' * max(1, 45 - len(label))}┐")
    print(f"  │  Allocated:       {stats['allocated_mb']:>8.1f} MB")
    print(f"  │  Reserved:        {stats['reserved_mb']:>8.1f} MB")
    print(f"  │  Peak allocated:  {stats['peak_allocated_mb']:>8.1f} MB")

    if "model_params_mb" in stats:
        print(f"  │  ── Breakdown ──")
        print(f"  │  Parameters:      {stats['model_params_mb']:>8.1f} MB")
        print(f"  │  Gradients:       {stats['gradients_mb']:>8.1f} MB")
        if "optimizer_states_mb" in stats:
            print(f"  │  Optimizer state: {stats['optimizer_states_mb']:>8.1f} MB")
        print(f"  │  Activations/etc: {stats['activations_and_other_mb']:>8.1f} MB")

    print(f"  └{'─' * 52}┘")


def memory_snapshot(model=None, optimizer=None, label="snapshot"):
    """Собрать и напечатать snapshot памяти — удобная обёртка."""
    stats = detailed_memory_stats(model, optimizer, label)
    print_memory_stats(stats)
    return stats
