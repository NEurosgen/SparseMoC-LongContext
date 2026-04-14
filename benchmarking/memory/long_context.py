
import gc
import os
import time
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from models.moc_ffn_triton.SparseSiLUFFN import SparseSiLUFFN
from torch.utils.data import Dataset
def free_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()



def load_model(model_path, sparse_weights_path = None, device='cuda'):
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype="auto", device_map=device,
        attn_implementation="sdpa"
    )
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





def try_training_step(model, tokenizer, seq_len, batch_size=1, device="cuda",
                      use_grad_checkpoint=True):
    free_gpu()
    try:
        model.train()
        if use_grad_checkpoint:
            model.gradient_checkpointing_enable()

        for p in model.parameters():
            p.requires_grad = True

        input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len), device=device)
        labels = input_ids.clone()
        torch.cuda.reset_peak_memory_stats()
        outputs = model(input_ids=input_ids, labels=labels)
        outputs.loss.backward()
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        model.zero_grad(set_to_none=True)
        del outputs, input_ids, labels
        free_gpu()

        if use_grad_checkpoint:
            model.gradient_checkpointing_disable()

        return round(peak_mb, 2)

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            model.zero_grad(set_to_none=True)
            free_gpu()
            if use_grad_checkpoint:
                try:
                    model.gradient_checkpointing_disable()
                except:
                    pass
            return None
        raise


def memory_sweep(model, model_name, tokenizer, seq_lens, device="cuda",
                 use_grad_checkpoint=True):
    """Sweep seq_len и замерить peak memory при training step."""
    print(f"\n  Memory Sweep: {model_name} (grad_checkpoint={use_grad_checkpoint})")
    print(f"  {'seq_len':>8} | {'peak_MB':>10} | {'status':>8}")
    print(f"  {'-'*32}")

    results = []
    for sl in seq_lens:
        peak = try_training_step(model, tokenizer, sl, device=device,
                                 use_grad_checkpoint=use_grad_checkpoint)
        if peak is not None:
            print(f"  {sl:>8} | {peak:>8.1f}MB | OK")
            results.append({"seq_len": sl, "peak_memory_mb": peak, "status": "ok"})
        else:
            print(f"  {sl:>8} | {'OOM':>10} | FAILED")
            results.append({"seq_len": sl, "peak_memory_mb": None, "status": "oom"})

    return results


def find_max_seq_len(model, model_name, tokenizer, lo=128, hi=16384,
                     device="cuda", use_grad_checkpoint=True):
    """Бинарный поиск максимального seq_len до OOM."""
    print(f"\n  Finding max seq_len: {model_name}")
    print(f"  Search range: [{lo}, {hi}]")

    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        mid = (mid // 64) * 64
        if mid < lo:
            break

        peak = try_training_step(model, tokenizer, mid, device=device,
                                 use_grad_checkpoint=use_grad_checkpoint)
        if peak is not None:
            print(f"  seq_len={mid}: OK ({peak:.0f}MB)")
            best = mid
            lo = mid + 64
        else:
            print(f"  seq_len={mid}: OOM")
            hi = mid - 64

    print(f"  → Max seq_len: {best}")
    return best

def create_pg19_dataloader(tokenizer, seq_len, batch_size=1, max_samples=200):
    """DataLoader из PG-19 для long-context training."""
    dataset = load_dataset("deepmind/pg19", split="train", streaming=True)

    all_ids = []
    for i, example in enumerate(dataset):
        if i >= max_samples:
            break
        tokens = tokenizer(example["text"], truncation=True, max_length=seq_len * 4,
                           return_attention_mask=False)["input_ids"]
        all_ids.extend(tokens)

    chunks = []
    for i in range(0, len(all_ids) - seq_len, seq_len):
        chunk = all_ids[i:i + seq_len]
        chunks.append({
            "input_ids": chunk,
            "attention_mask": [1] * seq_len,
            "labels": chunk,
        })

    

    class SimpleDataset(Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return {k: torch.tensor(v) for k, v in self.data[idx].items()}

    ds = SimpleDataset(chunks)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      collate_fn=default_data_collator, pin_memory=True)


def train_one_epoch(model, tokenizer, seq_len, lr=2e-5,
                    grad_accum_steps=4, max_steps=None,
                    device="cuda", use_grad_checkpoint=True):
    """
    Один epoch FT на PG-19 с gradient checkpointing + accumulation.
    Возвращает метрики обучения.
    """
    model.train()
    if use_grad_checkpoint:
        model.gradient_checkpointing_enable()

    for p in model.parameters():
        p.requires_grad = True

    dataloader = create_pg19_dataloader(tokenizer, seq_len, batch_size=1)
    print(f"  PG-19 dataloader: {len(dataloader)} batches, seq_len={seq_len}")

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    total_loss = 0.0
    total_tokens = 0
    step_count = 0
    log_every = max(1, len(dataloader) // 10)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    optimizer.zero_grad()
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss / grad_accum_steps
        loss.backward()

        total_loss += outputs.loss.item()
        total_tokens += (labels != -100).sum().item()

        if (batch_idx + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1

        if batch_idx % log_every == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(f"    batch {batch_idx}/{len(dataloader)} | loss: {avg_loss:.4f}")

        if max_steps and step_count >= max_steps:
            print(f"    Early stop at step {step_count}")
            break

    elapsed = time.time() - t0
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    avg_loss = total_loss / max(len(dataloader), 1)

    if use_grad_checkpoint:
        model.gradient_checkpointing_disable()

    return {
        "seq_len": seq_len,
        "avg_loss": round(avg_loss, 6),
        "total_tokens": total_tokens,
        "steps": step_count,
        "time_s": round(elapsed, 1),
        "tokens_per_sec": round(total_tokens / elapsed, 1) if elapsed > 0 else 0,
        "peak_memory_mb": round(peak_mb, 2),
    }



def main():
    model_path = "/home/eugen/MyDir/SHAD/Eff_ML/Project/llms/saved_dir/qwen3-0.6b"
 
    sparse_path = "/home/eugen/MyDir/SHAD/Eff_ML/Project/saved_dir/full_pipeline/sparse_ffn_weights.pt"
    os.makedirs("/home/eugen/MyDir/SHAD/Eff_ML/Project/log/long_ctx_eval", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_seq_len = [512, 1024, 2048, 4096, 8192]
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {vram_gb:.1f} GB")

    tokenizer = AutoTokenizer.from_pretrained(model_path)


    model = load_model(model_path,sparse_weights_path=sparse_path ,device=device)

    model_sweep = memory_sweep(model, "Qwen3", tokenizer,train_seq_len , device=device, use_grad_checkpoint=False)



    model_max_sl = find_max_seq_len(model, "Qwen3", tokenizer, device=device, use_grad_checkpoint=False)



    print(f"\n  Training (1 epoch PG-19, seq_len={train_seq_len})...")
    model_train = train_one_epoch(
        model, tokenizer, train_seq_len,
        grad_accum_steps=4,
        max_steps=50,
        device=device,
    )

    print(f"  → loss: {model_train['avg_loss']:.4f}, peak: {model_train['peak_memory_mb']:.0f}MB")

    free_model(model)


