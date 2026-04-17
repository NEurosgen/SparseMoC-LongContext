import gc
import json
import os
import time
from datetime import datetime

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, default_data_collator
from datasets import load_dataset

from utils import (
    load_model, free_model, free_gpu,
    detailed_memory_stats, print_memory_stats, memory_snapshot,
)

def try_training_step(model, tokenizer, seq_len, batch_size=1, device="cuda",
                      use_grad_checkpoint=True, use_amp=True):
    """
    Выполняет один forward+backward шаг и возвращает peak memory.
    Возвращает None при OOM.
    """
    free_gpu()
    try:
        model.train()
        if use_grad_checkpoint:
            model.gradient_checkpointing_enable()
        dummy_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

       

        for p in model.parameters():
            p.requires_grad = True

        input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len), device=device)
        labels = input_ids.clone()

        torch.cuda.reset_peak_memory_stats()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            outputs = model(input_ids=input_ids, labels=labels)
        outputs.loss.backward()

        dummy_optimizer.step()
        
        peak_mb = torch.cuda.max_memory_allocated()/(1024**2)

        dummy_optimizer.zero_grad(set_to_none=True)
        del outputs, input_ids, labels, dummy_optimizer
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
                except Exception:
                    pass
            return None
        raise


def memory_sweep(model, model_name, tokenizer, seq_lens, device="cuda",
                 use_grad_checkpoint=True, use_amp=True):
    """Sweep по seq_len и замер peak memory при training step."""
    print(f"\n  Memory Sweep: {model_name}"
          f" (grad_ckpt={use_grad_checkpoint}, amp={use_amp})")
    print(f"  {'seq_len':>8} | {'peak_MB':>10} | {'status':>8}")
    print(f"  {'-' * 35}")

    results = []
    for sl in seq_lens:
        peak = try_training_step(
            model, tokenizer, sl, device=device,
            use_grad_checkpoint=use_grad_checkpoint,
            use_amp=use_amp,
        )
        if peak is not None:
            print(f"  {sl:>8} | {peak:>8.1f}MB | OK")
            results.append({"seq_len": sl, "peak_memory_mb": peak, "status": "ok"})
        else:
            print(f"  {sl:>8} | {'OOM':>10} | FAILED")
            results.append({"seq_len": sl, "peak_memory_mb": None, "status": "oom"})

    return results


def find_max_seq_len(model, model_name, tokenizer, lo=128, hi=16384,
                     device="cuda", use_grad_checkpoint=True, use_amp=True):
    """Бинарный поиск максимального seq_len до OOM."""
    print(f"\n  Finding max seq_len: {model_name}")
    print(f"  Search range: [{lo}, {hi}]")

    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        mid = (mid // 64) * 64
        if mid < lo:
            break

        peak = try_training_step(
            model, tokenizer, mid, device=device,
            use_grad_checkpoint=use_grad_checkpoint,
            use_amp=use_amp,
        )
        if peak is not None:
            print(f"  seq_len={mid}: OK ({peak:.0f}MB)")
            best = mid
            lo = mid + 64
        else:
            print(f"  seq_len={mid}: OOM")
            hi = mid - 64

    print(f"  → Max seq_len: {best}")
    return best



class SimpleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {k: torch.tensor(v) for k, v in self.data[idx].items()}


def create_pg19_dataloader(tokenizer, seq_len, batch_size=1, max_samples=200):
    """DataLoader из PG-19 для long-context training."""
    dataset = load_dataset("deepmind/pg19", split="train", streaming=True)

    all_ids = []
    for i, example in enumerate(dataset):
        if i >= max_samples:
            break
        tokens = tokenizer(
            example["text"], truncation=True, max_length=seq_len * 4,
            return_attention_mask=False
        )["input_ids"]
        all_ids.extend(tokens)

    chunks = []
    for i in range(0, len(all_ids) - seq_len + 1, seq_len):
        chunk = all_ids[i:i + seq_len]
        chunks.append({
            "input_ids": chunk,
            "attention_mask": [1] * seq_len,
            "labels": chunk,
        })

    ds = SimpleDataset(chunks)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        collate_fn=default_data_collator, pin_memory=True,
    )


def train_one_epoch(model, tokenizer, seq_len, lr=2e-5,
                    grad_accum_steps=4, max_steps=None,
                    device="cuda", use_grad_checkpoint=True, use_amp=True):
    """
    Один epoch FT на PG-19 с gradient checkpointing + accumulation + AMP.
    Возвращает метрики обучения + подробный timeline памяти.
    """
    model.train()
    if use_grad_checkpoint:
        model.gradient_checkpointing_enable()

    for p in model.parameters():
        p.requires_grad = True

    dataloader = create_pg19_dataloader(tokenizer, seq_len, batch_size=1)
    print(f"  PG-19 dataloader: {len(dataloader)} batches, seq_len={seq_len}")

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    scaler = GradScaler(enabled=use_amp)

    memory_timeline = []
    torch.cuda.reset_peak_memory_stats()

    snap = memory_snapshot(model, optimizer, label="before_training")
    memory_timeline.append(snap)

    total_loss = 0.0
    total_tokens = 0
    step_count = 0
    log_every = max(1, len(dataloader) // 10)

    t0 = time.time()
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        if use_amp:
            with autocast(dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / grad_accum_steps
            scaler.scale(loss).backward()
        else:
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / grad_accum_steps
            loss.backward()

        total_loss += outputs.loss.item()
        total_tokens += (labels != -100).sum().item()

        if (batch_idx + 1) % grad_accum_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()
            step_count += 1

            if step_count % 10 == 0:
                snap = memory_snapshot(
                    model, optimizer,
                    label=f"step_{step_count}"
                )
                memory_timeline.append(snap)

        if batch_idx % log_every == 0:
            avg_loss = total_loss / (batch_idx + 1)
            cur_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(f"    batch {batch_idx}/{len(dataloader)} | "
                  f"loss: {avg_loss:.4f} | peak: {cur_peak:.0f}MB")

        if max_steps and step_count >= max_steps:
            print(f"    Early stop at step {step_count}")
            break

    elapsed = time.time() - t0
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    avg_loss = total_loss / max(batch_idx + 1, 1)

    snap = memory_snapshot(model, optimizer, label="after_training")
    memory_timeline.append(snap)

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
        "memory_timeline": memory_timeline,
    }
def main():
    model_path = "Qwen/Qwen3-0.6B"
    sparse_path = None 
    # sparse_path = "saved_dir/full_pipeline/sparse_ffn_weights.pt"

    log_dir = "log/long_ctx_eval"
    os.makedirs(log_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_seq_lens = [512, 1024, 2048, 4096, 8192, 16384 , 32768]

    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {vram_gb:.1f} GB")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model_name = "Qwen3-Sparse" if sparse_path else "Qwen3-Dense"
    model = load_model(model_path, sparse_weights_path=sparse_path, device=device)

    memory_snapshot(model, label=f"{model_name} loaded")

    print(f"\n{'=' * 60}")
    print(f"  PHASE 1: Memory Sweep — {model_name}")
    print(f"{'=' * 60}")
    sweep_results = memory_sweep(
        model, model_name, tokenizer, train_seq_lens,
        device=device, use_grad_checkpoint=True, use_amp=True,
    )

    print(f"\n{'=' * 60}")
    print(f"  PHASE 2: Max Seq Len — {model_name}")
    print(f"{'=' * 60}")
    max_sl = find_max_seq_len(
        model, model_name, tokenizer,
        device=device, use_grad_checkpoint=True, use_amp=True,
    )

    print(f"\n{'=' * 60}")
    print(f"  PHASE 3: Training — {model_name}")
    print(f"{'=' * 60}")

    training_results = []
    for sl in train_seq_lens:
        sweep_entry = next((r for r in sweep_results if r["seq_len"] == sl), None)
        if sweep_entry and sweep_entry["status"] == "oom":
            print(f"\n  Skipping seq_len={sl} (OOM in sweep)")
            continue

        print(f"\n  Training seq_len={sl}...")
        result = train_one_epoch(
            model, tokenizer, sl,
            grad_accum_steps=4,
            max_steps=50,
            device=device,
            use_grad_checkpoint=True,
            use_amp=True,
        )
        print(f"  → loss: {result['avg_loss']:.4f}, "
              f"peak: {result['peak_memory_mb']:.0f}MB, "
              f"tokens/s: {result['tokens_per_sec']:.0f}")
        training_results.append(result)


    all_results = {
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "gpu": torch.cuda.get_device_name() if device == "cuda" else "cpu",
        "vram_gb": round(vram_gb, 1) if device == "cuda" else 0,
        "sweep": sweep_results,
        "max_seq_len": max_sl,
        "training": training_results,
    }

    results_file = os.path.join(log_dir, f"{model_name}_results.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {results_file}")

    free_model(model)


if __name__ == "__main__":
    main()
