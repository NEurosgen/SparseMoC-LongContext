from fine_tuning.abstract_pipeline import Pipeline
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import time
import math


def unfreeze_ffn_layer(model):
    for i in range(len(model.model.layers)):
        for param in model.model.layers[i].mlp.parameters():
            param.requires_grad = True


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


class EndToEndFT(Pipeline):
    """
    End-to-end fine-tuning FFN-слоёв модели.

    """

    def __init__(self, config):
        super().__init__(config)

    def _evaluate(self, model, eval_dataloader):
        """Быстрая оценка loss на eval данных."""
        model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                input_ids = batch["input_ids"].to(model.device)
                labels = batch["labels"].to(model.device)
                outputs = model(input_ids=input_ids, labels=labels)
                active_tokens = (labels[:, 1:] != -100).sum().item()
                if active_tokens > 0:
                    total_loss += outputs.loss.item() * active_tokens
                    total_tokens += active_tokens

        model.train()
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
        ppl = math.exp(min(avg_loss, 700))
        return avg_loss, ppl

    def process(self, model, data, profiler=None):
        """
        data — train dataloader.
        """
        freeze_model(model)
        unfreeze_ffn_layer(model)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        total_trainable = sum(p.numel() for p in trainable_params)
        total_all = sum(p.numel() for p in model.parameters())
        
        self.logger.info(
            f'Trainable params: {total_trainable:,} / {total_all:,} '
            f'({100*total_trainable/total_all:.1f}%)'
        )
        self.logger.info(f'LR: {self.config.lr}, Epochs: {self.config.epochs}')

        optimizer = AdamW(params=trainable_params, lr=self.config.lr)
        
        total_batches = len(data)
        total_steps = total_batches * self.config.epochs
        warmup_ratio = getattr(self.config, 'warmup_ratio', 0.1)
        warmup_steps = int(total_steps * warmup_ratio)
        
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
        
        self.logger.info(
            f'Total steps: {total_steps}, Warmup: {warmup_steps}, '
            f'Batches/epoch: {total_batches}'
        )

        eval_dataloader = getattr(self.config, 'eval_dataloader', None)
        patience = getattr(self.config, 'patience', 2)
        best_eval_loss = float('inf')
        epochs_no_improve = 0
        
        best_loss = float('inf')
        loss_history = []
        eval_history = []
        global_step = 0

        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            epoch_loss = 0.0
            num_batches = 0
            model.train()

            for batch_idx, batch in enumerate(data):
                optimizer.zero_grad()
                
                input_ids = batch["input_ids"].to(model.device)
                labels = batch["labels"].to(model.device)
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss

                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                
                if global_step < warmup_steps:
                    warmup_factor = (global_step + 1) / warmup_steps
                    for pg in optimizer.param_groups:
                        pg['lr'] = self.config.lr * warmup_factor
                else:
                    scheduler.step()

                global_step += 1
                batch_loss = loss.item()
                epoch_loss += batch_loss
                num_batches += 1

                if profiler:
                    profiler.step()

                if batch_idx % 10 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    self.logger.info(
                        f'Epoch {epoch}/{self.config.epochs-1} | '
                        f'Batch {batch_idx}/{total_batches} | '
                        f'Loss: {batch_loss:.6f} | '
                        f'Grad norm: {grad_norm:.4f} | '
                        f'LR: {current_lr:.2e}'
                    )
                    self.log_metric(
                        stage='finetune',
                        epoch=epoch, batch=batch_idx,
                        loss=round(batch_loss, 6),
                        grad_norm=round(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm, 4),
                        lr=current_lr,
                    )

                if profiler and batch_idx > 15:
                    self.logger.info('Profiler mode: stopping early after 15 batches')
                    return {
                        'final_avg_loss': epoch_loss / max(num_batches, 1),
                        'total_epochs': 0,
                        'early_stop': True,
                    }

            epoch_time = time.time() - epoch_start
            epoch_avg = epoch_loss / max(num_batches, 1)
            loss_history.append(epoch_avg)
            
            if epoch_avg < best_loss:
                best_loss = epoch_avg

            eval_info = ""
            if eval_dataloader is not None:
                eval_loss, eval_ppl = self._evaluate(model, eval_dataloader)
                eval_history.append({'epoch': epoch, 'eval_loss': eval_loss, 'eval_ppl': eval_ppl})
                eval_info = f', eval_loss={eval_loss:.6f}, eval_ppl={eval_ppl:.2f}'

                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    self.logger.info(
                        f'Early stopping at epoch {epoch}: '
                        f'eval_loss not improved for {patience} epochs'
                    )
                    break

            self.logger.info(
                f'--- Epoch {epoch} summary: '
                f'train_avg={epoch_avg:.6f}, best={best_loss:.6f}, '
                f'time={epoch_time:.1f}s{eval_info} ---'
            )
            self.log_metric(
                stage='finetune_epoch',
                epoch=epoch,
                train_avg_loss=round(epoch_avg, 6),
                best_loss=round(best_loss, 6),
                eval_loss=round(eval_loss, 6) if eval_dataloader else None,
                eval_ppl=round(eval_ppl, 2) if eval_dataloader else None,
                epoch_time_s=round(epoch_time, 1),
                lr=optimizer.param_groups[0]['lr'],
            )

        return {
            'final_avg_loss': epoch_avg,
            'best_loss': best_loss,
            'total_epochs': epoch + 1,
            'loss_history': loss_history,
            'eval_history': eval_history,
        }

    def log_result(self, result):
        if result is None:
            self.logger.warning('Fine-tuning returned None')
            return
        
        self.logger.info(
            f'Fine-tuning completed | '
            f'Final avg loss: {result["final_avg_loss"]:.6f} | '
            f'Best loss: {result.get("best_loss", "N/A")} | '
            f'Epochs: {result["total_epochs"]}'
        )
        if 'loss_history' in result:
            self.logger.info(f'Loss history: {[f"{l:.4f}" for l in result["loss_history"]]}')
        if result.get('eval_history'):
            for eh in result['eval_history']:
                self.logger.info(f'  Eval epoch {eh["epoch"]}: loss={eh["eval_loss"]:.4f}, ppl={eh["eval_ppl"]:.2f}')
