
import math
import os
import time
from datetime import datetime

import torch
from transformers import AutoTokenizer, default_data_collator
from datasets import load_dataset
from torch.utils.data import DataLoader

from utils import load_model, free_model



def create_eval_dataloader(tokenizer, seq_len=512, batch_size=1):
    """DataLoader из wikitext-2-raw-v1 (test split)."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    dataset = dataset.filter(lambda x: len(x["text"]) > 0)

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=seq_len)

    tokenized = dataset.map(tokenize_fn, batched=True, num_proc=4, remove_columns=["text"])

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total = len(concatenated[list(examples.keys())[0]])
        total = (total // seq_len) * seq_len
        result = {
            k: [t[i:i + seq_len] for i in range(0, total, seq_len)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_dataset = tokenized.map(group_texts, batched=True, num_proc=4)
    return DataLoader(
        lm_dataset, shuffle=False, batch_size=batch_size,
        collate_fn=default_data_collator, pin_memory=True,
    )


def evaluate_perplexity(model, dataloader, device="cuda"):
    """Вычисляет perplexity модели на данном DataLoader."""
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, labels=labels)
            active_tokens = (labels[:, 1:] != -100).sum().item()
            if active_tokens > 0:
                total_loss += outputs.loss.item() * active_tokens
                total_tokens += active_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    ppl = math.exp(min(avg_loss, 700))
    return {"perplexity": round(ppl, 4), "avg_loss": round(avg_loss, 6), "tokens": total_tokens}


def run_lm_eval(model, tokenizer, tasks=("arc_easy", "hellaswag", "winogrande"),
                batch_size=4, device="cuda"):
    """Запускает lm-eval-harness бенчмарки на загруженной модели."""
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size, device=device)

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=list(tasks),
        batch_size=batch_size,
    )

    summary = {}
    for task_name, task_data in results.get("results", {}).items():
        metric_key = None
        for k in ("acc,none", "acc_norm,none", "acc"):
            if k in task_data:
                metric_key = k
                break
        if metric_key:
            summary[task_name] = round(task_data[metric_key], 4)
        else:
            for k, v in task_data.items():
                if isinstance(v, (int, float)) and not k.startswith("alias"):
                    summary[task_name] = round(v, 4)
                    break

    return summary


SAMPLE_PROMPTS = [
    "The theory of relativity states that",
    "In machine learning, transformers are",
    "The capital of France is",
    "Once upon a time, in a distant galaxy,",
    "def fibonacci(n):",
]


def generate_samples(model, tokenizer, prompts=None, max_new_tokens=100, device="cuda"):
    """Генерирует тексты для визуального сравнения."""
    if prompts is None:
        prompts = SAMPLE_PROMPTS

    results = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )
        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        results.append({"prompt": prompt, "generated": full_text})

    return results



def evaluate_single_model(model_name, model, tokenizer, dataloader,
                          run_harness=True, run_gen=True, device="cuda"):
    """Полная оценка одной модели."""
    print(f"\n{'='*60}")
    print(f"  Evaluating: {model_name}")
    print(f"{'='*60}")

    result = {"model": model_name}

    print(f"  [1/3] Perplexity...")
    t0 = time.time()
    ppl_result = evaluate_perplexity(model, dataloader, device)
    result["perplexity"] = ppl_result
    print(f"    PPL: {ppl_result['perplexity']:.4f}  ({time.time()-t0:.1f}s)")

    if run_harness:
        print(f"  [2/3] Zero-shot benchmarks (lm-eval)...")
        t0 = time.time()
        try:
            harness_result = run_lm_eval(model, tokenizer, device=device)
            result["lm_eval"] = harness_result
            for task, acc in harness_result.items():
                print(f"    {task}: {acc:.4f}")
            print(f"    ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"    ⚠ lm-eval failed: {e}")
            result["lm_eval"] = {"error": str(e)}
    else:
        print(f"  [2/3] lm-eval SKIPPED")

    if run_gen:
        print(f"  [3/3] Text generation...")
        gen_result = generate_samples(model, tokenizer, device=device)
        result["generation"] = gen_result
        for g in gen_result[:2]:
            print(f"    Prompt: {g['prompt'][:50]}...")
            print(f"    Output: {g['generated'][:80]}...")
    else:
        print(f"  [3/3] Generation SKIPPED")

    return result


def main():

    model_path = "/home/eugen/MyDir/SHAD/Eff_ML/Project/llms/saved_dir/qwen3-0.6b"
    sparse_wights = "/home/eugen/MyDir/SHAD/Eff_ML/Project/saved_dir/full_pipeline_512/sparse_ffn_weights.pt"
    seq_len = 512
    batch_size = 1


    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    tokenizer = AutoTokenizer.from_pretrained(model_path)


    print("Preparing eval data...")
    dataloader = create_eval_dataloader(tokenizer, seq_len=seq_len, batch_size=batch_size)
    print(f"  {len(dataloader)} batches")


    model = load_model(model_path,sparse_wights ,device)
    free_model(model)






if __name__ == "__main__":
    main()
