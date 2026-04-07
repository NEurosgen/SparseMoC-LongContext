from pipeline.abstract_pipe import Pipeline
from torch.optim import AdamW
import torch
import time


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
    
    Подробное логирование:
    - Per-batch: loss, grad_norm
    - Per-epoch: avg_loss, epoch_time, lr
    - Итог: финальный avg_loss, trainable params
    """

    def __init__(self, config):
        super().__init__(config)

    def process(self, model, data, profiler=None):
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
        self.logger.info(f'Total batches per epoch: {total_batches}')
        
        best_loss = float('inf')
        loss_history = []

        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(data):
                optimizer.zero_grad()
                
                input_ids = batch["input_ids"].to(model.device)
                labels = batch["labels"].to(model.device)
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss

                loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                
                optimizer.step()
                
                batch_loss = loss.item()
                epoch_loss += batch_loss
                num_batches += 1

                if profiler:
                    profiler.step()

                if batch_idx % 10 == 0:
                    self.logger.info(
                        f'Epoch {epoch}/{self.config.epochs-1} | '
                        f'Batch {batch_idx}/{total_batches} | '
                        f'Loss: {batch_loss:.6f} | '
                        f'Grad norm: {grad_norm:.4f}'
                    )
                    self.log_metric(
                        stage='finetune',
                        epoch=epoch,
                        batch=batch_idx,
                        loss=round(batch_loss, 6),
                        grad_norm=round(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm, 4),
                        lr=self.config.lr,
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

            self.logger.info(
                f'--- Epoch {epoch} summary: '
                f'avg_loss={epoch_avg:.6f}, '
                f'best_loss={best_loss:.6f}, '
                f'time={epoch_time:.1f}s, '
                f'batches={num_batches} ---'
            )
            self.log_metric(
                stage='finetune_epoch',
                epoch=epoch,
                avg_loss=round(epoch_avg, 6),
                best_loss=round(best_loss, 6),
                epoch_time_s=round(epoch_time, 1),
                num_batches=num_batches,
            )

        return {
            'final_avg_loss': epoch_avg,
            'best_loss': best_loss,
            'total_epochs': self.config.epochs,
            'loss_history': loss_history,
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
