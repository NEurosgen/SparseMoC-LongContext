import torch
import time
import json
import os
import logging
from dataclasses import dataclass, asdict
from datetime import datetime

from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from datasets import load_dataset
from torch.utils.data import DataLoader

from pipeline.distilation import (
    Distilation, save_sparse_model, load_sparse_model, unwrap_distillation
)
from pipeline.perplexity import PerplexEst
from pipeline.end_to_end_ft import EndToEndFT


@dataclass
class FullPipelineConfig:
    """Единая конфигурация для полного pipeline."""
    base_model_path: str = "/home/eugen/MyDir/SHAD/Eff_ML/Project/llms/saved_dir/qwen3-0.6b"
    sparse_weights_path: str = ""  # если пусто то запускаем дистилляцию с нуля
    
    top_k: int = 128
    lr_distill: float = 1e-4
    epochs_distill: int = 3
    
    lr_ft: float = 2e-5
    epochs_ft: int = 3
    
    seq_len: int = 512
    batch_size_train: int = 2
    batch_size_eval: int = 1
    num_workers: int = 4
    
    save_dir: str = "/home/eugen/MyDir/SHAD/Eff_ML/Project/saved_dir/full_pipeline"
    log_dir: str = "/home/eugen/MyDir/SHAD/Eff_ML/Project/log/full_pipeline"
    
    # Опционально
    run_distillation: bool = True
    run_finetuning: bool = True



def create_wikitext_dataloader(tokenizer, split, seq_len, batch_size, num_workers=4, shuffle=False):
    """Создание dataloader из wikitext-2"""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    dataset = dataset.filter(lambda x: len(x["text"]) > 0)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=seq_len)

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_workers,
        remove_columns=["text"]
    )

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        if total_length >= seq_len:
            total_length = (total_length // seq_len) * seq_len

        result = {
            k: [t[i: i + seq_len] for i in range(0, total_length, seq_len)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_dataset = tokenized.map(group_texts, batched=True, num_proc=num_workers)

    return DataLoader(
        lm_dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        collate_fn=default_data_collator,
        pin_memory=True,
    )


def run_full_pipeline(config: FullPipelineConfig):
    """
    Полный pipeline:
        1. Загрузка модели
        2. Дистилляция (опционально)
        3. Перплексия (после дистилляции)
        4. Fine-tuning (опционально)
        5. Перплексия (после FT)
        6. Сохранение модели
        7. Итоговый отчёт
    """
    logger = logging.getLogger('FullPipeline')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter(
            '[%(asctime)s] %(name)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        
        os.makedirs(config.log_dir, exist_ok=True)
        fh = logging.FileHandler(
            os.path.join(config.log_dir, 'full_pipeline.log'), encoding='utf-8'
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    pipeline_start = time.time()
    report = {
        'config': asdict(config),
        'started_at': datetime.now().isoformat(),
        'stages': {},
    }

    logger.info('=' * 70)
    logger.info('FULL PIPELINE START')
    logger.info('=' * 70)
    logger.info(f'Config: {json.dumps(asdict(config), indent=2, ensure_ascii=False)}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Device: {device}')
    if device == 'cuda':
        logger.info(f'GPU: {torch.cuda.get_device_name()}')
        logger.info(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

   
    logger.info('--- Stage 1: Loading model & tokenizer ---')
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_path)

    if config.sparse_weights_path and os.path.exists(config.sparse_weights_path):
        logger.info(f'Loading sparse model from {config.sparse_weights_path}')
        model = load_sparse_model(config.base_model_path, config.sparse_weights_path, device=device)
        config.run_distillation = False  # уже sparse, дистилляция не нужна
        logger.info('Sparse model loaded, skipping distillation')
    else:
        logger.info(f'Loading base model from {config.base_model_path}')
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_path, torch_dtype="auto", device_map=device
        )

    load_time = time.time() - t0
    logger.info(f'Model loaded in {load_time:.1f}s')
    report['stages']['load_model'] = {'time_s': round(load_time, 1)}


    logger.info('--- Stage 2: Preparing data ---')
    t0 = time.time()

    train_loader = create_wikitext_dataloader(
        tokenizer, split='train',
        seq_len=config.seq_len,
        batch_size=config.batch_size_train,
        num_workers=config.num_workers,
        shuffle=True,
    )
    eval_loader = create_wikitext_dataloader(
        tokenizer, split='test',
        seq_len=config.seq_len,
        batch_size=config.batch_size_eval,
        num_workers=config.num_workers,
        shuffle=False,
    )

    data_time = time.time() - t0
    logger.info(f'Data prepared in {data_time:.1f}s | train: {len(train_loader)} batches, eval: {len(eval_loader)} batches')
    report['stages']['data_prep'] = {
        'time_s': round(data_time, 1),
        'train_batches': len(train_loader),
        'eval_batches': len(eval_loader),
    }


    if config.run_distillation:
        logger.info('--- Stage 3: Distillation ---')

        @dataclass
        class DistillConfig:
            top_k: int = config.top_k
            lr: float = config.lr_distill
            epochs: int = config.epochs_distill
            save_dir: str = config.save_dir
            log_dir: str = config.log_dir

        distill_pipe = Distilation(DistillConfig())
        distill_result = distill_pipe.run_process(model, train_loader)
        model = distill_result['model']

        report['stages']['distillation'] = {
            'final_avg_loss': distill_result['final_avg_loss'],
            'epochs': distill_result['total_epochs'],
        }
    else:
        logger.info('--- Stage 3: Distillation SKIPPED ---')


    logger.info('--- Stage 4: Perplexity (post-distillation) ---')

    @dataclass
    class EvalConfig:
        log_dir: str = config.log_dir

    ppl_pipe_before = PerplexEst(EvalConfig())
    ppl_result_before = ppl_pipe_before.run_process(model, eval_loader)
    ppl_before = ppl_result_before['perplexity']

    report['stages']['perplexity_post_distill'] = {
        'perplexity': round(ppl_before, 4),
        'avg_loss': round(ppl_result_before['avg_loss'], 6),
        'tokens': ppl_result_before['total_tokens'],
    }
    logger.info(f'Perplexity after distillation: {ppl_before:.4f}')

    if config.run_finetuning:
        logger.info('--- Stage 5: Fine-tuning ---')

        @dataclass
        class FTConfig:
            lr: float = config.lr_ft
            epochs: int = config.epochs_ft
            log_dir: str = config.log_dir

        ft_pipe = EndToEndFT(FTConfig())
        ft_result = ft_pipe.run_process(model, train_loader)

        report['stages']['finetuning'] = {
            'final_avg_loss': ft_result['final_avg_loss'],
            'best_loss': ft_result.get('best_loss'),
            'epochs': ft_result['total_epochs'],
            'loss_history': ft_result.get('loss_history'),
        }
    else:
        logger.info('--- Stage 5: Fine-tuning SKIPPED ---')


    logger.info('--- Stage 6: Perplexity (post-finetuning) ---')

    ppl_pipe_after = PerplexEst(EvalConfig())
    ppl_result_after = ppl_pipe_after.run_process(model, eval_loader)
    ppl_after = ppl_result_after['perplexity']

    report['stages']['perplexity_post_ft'] = {
        'perplexity': round(ppl_after, 4),
        'avg_loss': round(ppl_result_after['avg_loss'], 6),
        'tokens': ppl_result_after['total_tokens'],
    }
    logger.info(f'Perplexity after fine-tuning: {ppl_after:.4f}')

    logger.info('--- Stage 7: Saving model ---')
    t0 = time.time()

    os.makedirs(config.save_dir, exist_ok=True)
    weights_path, num_saved = save_sparse_model(model, config.save_dir)
    save_time = time.time() - t0

    logger.info(f'Model saved to {weights_path} ({num_saved} layers) in {save_time:.1f}s')
    report['stages']['save_model'] = {
        'path': weights_path,
        'layers_saved': num_saved,
        'time_s': round(save_time, 1),
    }


    total_time = time.time() - pipeline_start
    report['total_time_s'] = round(total_time, 1)
    report['finished_at'] = datetime.now().isoformat()

    # Вычисление дельта перплексии
    ppl_delta = ppl_after - ppl_before
    ppl_delta_pct = (ppl_delta / ppl_before) * 100 if ppl_before > 0 else 0

    logger.info('=' * 70)
    logger.info('FULL PIPELINE REPORT')
    logger.info('=' * 70)
    logger.info(f'Total time:        {total_time:.1f}s ({total_time/60:.1f}min)')
    logger.info(f'PPL post-distill:  {ppl_before:.4f}')
    logger.info(f'PPL post-FT:       {ppl_after:.4f}')
    logger.info(f'PPL delta:         {ppl_delta:+.4f} ({ppl_delta_pct:+.1f}%)')
    logger.info(f'Model saved to:    {config.save_dir}')
    logger.info('=' * 70)


    report_path = os.path.join(config.log_dir, 'pipeline_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f'Full report saved to {report_path}')

    return report


# ============================================================
#  Entry point
# ============================================================

if __name__ == "__main__":
    config = FullPipelineConfig(
        # --- Модель ---
        base_model_path="/home/eugen/MyDir/SHAD/Eff_ML/Project/llms/saved_dir/qwen3-0.6b",
        sparse_weights_path="",  # пусто = дистилляция с нуля
        
        # --- Дистилляция ---
        top_k=128,
        lr_distill=1e-4,
        epochs_distill=3,
        
        # --- Fine-tuning ---
        lr_ft=2e-5,
        epochs_ft=3,
        
        # --- Данные ---
        seq_len=512,
        batch_size_train=2,
        batch_size_eval=1,
        
        # --- Сохранение ---
        save_dir="/home/eugen/MyDir/SHAD/Eff_ML/Project/saved_dir/full_pipeline",
        log_dir="/home/eugen/MyDir/SHAD/Eff_ML/Project/log/full_pipeline",
        
        # --- Этапы ---
        run_distillation=True,
        run_finetuning=True,
    )

    report = run_full_pipeline(config)
