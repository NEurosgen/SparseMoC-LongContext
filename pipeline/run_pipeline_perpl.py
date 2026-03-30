from pipeline.perplexity import PerplexEst
from pipeline.distilation import load_sparse_model
from transformers import AutoTokenizer, default_data_collator
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import os
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
def create_wikitext_dataloader(tokenizer, seq_len=1400, batch_size=4, num_workers=4):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    dataset = dataset.filter(lambda x: len(x["text"]) > 0)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=seq_len)

    tokenized_datasets = dataset.map(
        tokenize_function, 
        batched=True, 
        num_proc=num_workers, 
        remove_columns=["text"]
    )
    
    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= seq_len:
            total_length = (total_length // seq_len) * seq_len
        
        result = {
            k: [t[i : i + seq_len] for i in range(0, total_length, seq_len)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts, 
        batched=True, 
        num_proc=num_workers
    )
    
    dataloader = DataLoader(
        lm_datasets,
        shuffle=False, 
        batch_size=batch_size,
        collate_fn=default_data_collator,
        pin_memory=True
    )
    
    return dataloader

if __name__ == "__main__":
    base_model_path = "/home/eugen/MyDir/SHAD/Eff_ML/Project/llms/saved_dir/qwen3-0.6b"
    sparse_weights_path = "/home/eugen/MyDir/SHAD/Eff_ML/Project/saved_dir/sparse_distilled/sparse_ffn_weights.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading tokenizer from {base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    print(f"Creating dataloader for wikitext-2-raw-v1 (test split)...")
    dataloader = create_wikitext_dataloader(tokenizer, batch_size=1)
    
    print(f"Loading sparse model from weights: {sparse_weights_path}...")
    model = load_sparse_model(base_model_path, sparse_weights_path, device=device)
    # local_path = "/home/eugen/MyDir/SHAD/Eff_ML/Project/llms/saved_dir/qwen3-0.6b"
    # device = 'cuda'
    # model = AutoModelForCausalLM.from_pretrained(local_path, dtype="auto", device_map=device)
    class EvalConfig:
        def __init__(self, device):
            self.device = device

    config = EvalConfig(device)
    perplexity_pipeline = PerplexEst(config)
    
    print("Starting Perplexity evaluation...")
    perplexity = perplexity_pipeline.run_process(model, dataloader)
    
    print("\n" + "="*30)
    print(f"RESULT PERPLEXITY: {perplexity:.4f}")
    print("="*30)
