import math
import time
import torch
from pipeline.abstract_pipe import Pipeline


class PerplexEst(Pipeline):
    """
    Оценка перплексии модели на eval-данных.
    
    Подробное логирование:
    - Прогресс каждые N батчей (running avg loss, tokens processed)
    - Итог: avg_loss, perplexity, total_tokens, total_batches, время
    """

    def __init__(self, config):
        super().__init__(config)

    def process(self, model, data, profiler=None):
        """Оценка перплексии. Возвращает dict с метриками."""
        model.eval()

        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        total_batches = len(data)
        log_every = max(1, total_batches // 10)

        self.logger.info(f'Starting perplexity evaluation | {total_batches} batches')

        eval_start = time.time()

        with torch.no_grad():
            for batch_idx, batch in enumerate(data):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = model(input_ids=input_ids, labels=labels)
                active_tokens = (labels[:, 1:] != -100).sum().item()

                if active_tokens > 0:
                    total_loss += outputs.loss.item() * active_tokens
                    total_tokens += active_tokens

                num_batches += 1

                # Промежуточный лог
                if batch_idx % log_every == 0 and total_tokens > 0:
                    running_avg = total_loss / total_tokens
                    running_ppl = math.exp(min(running_avg, 700))
                    self.logger.info(
                        f'Batch {batch_idx}/{total_batches} | '
                        f'Tokens: {total_tokens:,} | '
                        f'Running avg loss: {running_avg:.4f} | '
                        f'Running PPL: {running_ppl:.2f}'
                    )

                if profiler:
                    profiler.step()

        eval_time = time.time() - eval_start

        avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
        perplexity = math.exp(min(avg_loss, 700))

        self.log_metric(
            stage='perplexity_eval',
            avg_loss=round(avg_loss, 6),
            perplexity=round(perplexity, 4),
            total_tokens=total_tokens,
            total_batches=num_batches,
            eval_time_s=round(eval_time, 1),
        )

        self.logger.info(
            f'Evaluation done in {eval_time:.1f}s | '
            f'Tokens: {total_tokens:,} | '
            f'Avg loss: {avg_loss:.4f} | '
            f'Perplexity: {perplexity:.4f}'
        )

        return {
            'perplexity': perplexity,
            'avg_loss': avg_loss,
            'total_tokens': total_tokens,
            'total_batches': num_batches,
            'eval_time_s': eval_time,
        }

    def log_result(self, result):
        if result is None:
            self.logger.warning('Perplexity estimation returned None')
            return

        self.logger.info(
            f'PERPLEXITY RESULT: {result["perplexity"]:.4f} | '
            f'Avg loss: {result["avg_loss"]:.4f} | '
            f'Tokens: {result["total_tokens"]:,} | '
            f'Time: {result["eval_time_s"]:.1f}s'
        )