from pipeline.end_to_end_ft import EndToEndFT
from pipeline.distilation import load_sparse_model
from transformers import AutoTokenizer, default_data_collator
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import os
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from torch.profiler import profile, record_function, ProfilerActivity, schedule
def create_wikitext_train_dataloader(tokenizer, seq_len=1400, batch_size=4, num_workers=4):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train") 
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
        shuffle=True,
        batch_size=batch_size,
        collate_fn=default_data_collator,
        pin_memory=True
    )
    
    return dataloader


if __name__ == "__main__":
    base_model_path = "/home/eugen/MyDir/SHAD/Eff_ML/Project/llms/saved_dir/qwen3-0.6b"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    dataloader = create_wikitext_train_dataloader(tokenizer,seq_len = 500 ,batch_size=2)
    
    print(f"Loading base model from {base_model_path}...")
    #model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype="auto", device_map=device)

    sparse_weights_path = "/home/eugen/MyDir/SHAD/Eff_ML/Project/saved_dir/sparse_distilled/sparse_ffn_weights.pt"
    model = load_sparse_model(base_model_path, sparse_weights_path, device=device)

    my_schedule = schedule(wait=2, warmup=3, active=5, repeat=1)

        
    class FTConfig:
        def __init__(self, lr, epochs):
            self.lr = lr
            self.epochs = epochs

    config = FTConfig(lr=2e-5, epochs=3)
    ft_pipeline = EndToEndFT(config)
    torch.cuda.memory._record_memory_history(
       max_entries=100000
   )
    print("Starting End-to-End Fine-Tuning...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=my_schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler_results'),
        record_shapes=True,
        with_stack=True,
        profile_memory=True
    ) as prof:
        ft_pipeline.run_process(model, dataloader, profiler=prof)
    torch.cuda.memory._dump_snapshot(f"log/model_mem.pickle")



    torch.cuda.memory._record_memory_history(enabled=None)
    
    print("\n" + "="*30)
    print("FINE-TUNING COMPLETED")
    print("="*30)