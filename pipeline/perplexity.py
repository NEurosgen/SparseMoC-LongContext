import math
import torch
from pipeline.abstract_pipe import Pipeline
import os
class PerplexEst(Pipeline):
    def __init__(self, config):
        super().__init__(config)

    def process(self, model, data):
        """Оценка перплексии у модели."""
        model.eval()

        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch in data:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = model(input_ids=input_ids, labels=labels)
                active_tokens = (labels[:, 1:] != -100).sum().item()

                if active_tokens > 0:
                    total_loss += outputs.loss.item()*active_tokens
                    total_tokens += active_tokens

        avg_loss = total_loss/total_tokens if total_tokens > 0 else float("inf")
        
        perplexity = math.exp(min(avg_loss, 700))
        
        return perplexity
    def log_result(self, result):
        log_dir = getattr(self.config, 'log_dir', './logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "perplexity_log.txt")
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"Estimated Perplexity: {result:.4f}\n")