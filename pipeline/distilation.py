from pipeline.abstract_pipe import Pipeline
import torch
from transformers import AutoModelForCausalLM
from torch import nn
import torch.nn.functional as F
import os
import time

from models.moc_ff_triton import SparseSiLUFFN
from torch.optim import AdamW


def load_sparse_model(model_path, sparse_weights_path, device='cuda'):
    """Загрузка модели с sparse FFN весами."""
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype="auto", device_map=device
    )
    
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


class FFNDistillationWrapper(nn.Module):
    def __init__(self, old_mlp, new_mlp):
        super().__init__()
        self.old_mlp = old_mlp
        self.new_mlp = new_mlp
        self.distill_loss = torch.tensor(0.0, device=old_mlp.gate_proj.weight.device)
        
        for param in self.old_mlp.parameters():
            param.requires_grad = False
            
        for param in self.new_mlp.parameters():
            param.requires_grad = True

    def forward(self, x):
        with torch.no_grad():
            target_out = self.old_mlp(x)
            x_detached = x.detach()

        with torch.enable_grad():
            pred_out = self.new_mlp(x_detached)
            self.distill_loss = F.mse_loss(pred_out, target_out)
        
        return target_out


def prepare_model_dist(model, top_k):
    model_dtype = model.dtype
    for param in model.parameters():
        param.requires_grad = False
    for i in range(len(model.model.layers)):
        old_mlp = model.model.layers[i].mlp
        d_model = old_mlp.gate_proj.in_features
        dffn = old_mlp.gate_proj.out_features
        new_mlp = SparseSiLUFFN(d_model=d_model, d_ffn=dffn, top_k=top_k)
        new_mlp = new_mlp.to(device=model.device, dtype=model_dtype)
        
        with torch.no_grad():
            new_mlp.w_gate.copy_(old_mlp.gate_proj.weight.t().to(model_dtype))
            new_mlp.w_up.copy_(old_mlp.up_proj.weight.t().to(model_dtype))
            new_mlp.w_down.copy_(old_mlp.down_proj.weight.t().to(model_dtype))
            
        wrapper = FFNDistillationWrapper(old_mlp, new_mlp)
        model.model.layers[i].mlp = wrapper


def unwrap_distillation(model):
    for i in range(len(model.model.layers)):
        wrapper = model.model.layers[i].mlp
        if isinstance(wrapper, FFNDistillationWrapper):
            model.model.layers[i].mlp = wrapper.new_mlp
    return model


def save_sparse_model(model, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    sparse_weights = {}
    for i, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        if isinstance(mlp, FFNDistillationWrapper):
            sparse_mlp = mlp.new_mlp
        elif isinstance(mlp, SparseSiLUFFN):
            sparse_mlp = mlp
        else:
            continue
        
        sparse_weights[f"layer_{i}"] = {
            "w_gate": sparse_mlp.w_gate.data.cpu(),
            "w_up": sparse_mlp.w_up.data.cpu(),
            "w_down": sparse_mlp.w_down.data.cpu(),
            "top_k": sparse_mlp.top_k,
            "d_model": sparse_mlp.d_model,
            "d_ffn": sparse_mlp.d_ffn,
        }
    
    weights_path = os.path.join(save_dir, "sparse_ffn_weights.pt")
    torch.save(sparse_weights, weights_path)
    return weights_path, len(sparse_weights)


class Distilation(Pipeline):
    """
    Дистилляция FFN-слоёв модели в SparseSiLUFFN.
    
    Подробное логирование:
    - Per-batch: total_loss, avg_layer_loss, min/max layer loss
    - Per-epoch: avg_loss, epoch_time, learning_rate
    - Итог: финальный loss, время, кол-во слоёв
    """

    def __init__(self, config):
        super().__init__(config=config)

    def process(self, model, data, profiler=None):
        """
        В качестве data приходит dataloader.
        Возвращает dict с моделью и историей метрик.
        """
        num_layers = len(model.model.layers)
        self.logger.info(f'Model layers: {num_layers}, top_k: {self.config.top_k}')
        self.logger.info(f'LR: {self.config.lr}, Epochs: {self.config.epochs}')
        
        prepare_model_dist(model, self.config.top_k)
        self.logger.info('Model wrapped with FFNDistillationWrapper')

        trainable_params = []
        for layer in model.model.layers:
            trainable_params.extend(list(layer.mlp.new_mlp.parameters()))
        
        total_trainable = sum(p.numel() for p in trainable_params)
        self.logger.info(f'Trainable parameters: {total_trainable:,}')
        
        optimizer = AdamW(trainable_params, lr=self.config.lr)
        model.eval()
        
        total_batches = len(data)
        self.logger.info(f'Total batches per epoch: {total_batches}')

        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(data):
                optimizer.zero_grad()
                
                input_ids = batch["input_ids"].to(model.device)
                
                with torch.no_grad():
                    _ = model(input_ids)
                
                # Собираем loss со всех слоёв
                layer_losses = []
                total_loss = torch.tensor(0.0, device=model.device)
                for layer in model.model.layers:
                    l = layer.mlp.distill_loss
                    layer_losses.append(l.item())
                    total_loss = total_loss + l
                
                total_loss.backward()
                optimizer.step()
                
                batch_loss = total_loss.item()
                epoch_loss += batch_loss
                num_batches += 1

                if batch_idx % 10 == 0:
                    avg_layer = batch_loss / num_layers
                    min_layer = min(layer_losses)
                    max_layer = max(layer_losses)
                    
                    self.logger.info(
                        f'Epoch {epoch}/{self.config.epochs-1} | '
                        f'Batch {batch_idx}/{total_batches} | '
                        f'Loss: {batch_loss:.6f} | '
                        f'Avg/layer: {avg_layer:.6f} | '
                        f'Min/Max layer: {min_layer:.6f}/{max_layer:.6f}'
                    )
                    
                    self.log_metric(
                        stage='distillation',
                        epoch=epoch,
                        batch=batch_idx,
                        total_loss=round(batch_loss, 6),
                        avg_layer_loss=round(avg_layer, 6),
                        min_layer_loss=round(min_layer, 6),
                        max_layer_loss=round(max_layer, 6),
                        lr=self.config.lr,
                    )

                if profiler:
                    profiler.step()

            epoch_time = time.time() - epoch_start
            epoch_avg = epoch_loss / max(num_batches, 1)
            
            self.logger.info(
                f'--- Epoch {epoch} summary: '
                f'avg_loss={epoch_avg:.6f}, '
                f'time={epoch_time:.1f}s, '
                f'batches={num_batches} ---'
            )
            self.log_metric(
                stage='distillation_epoch',
                epoch=epoch,
                avg_loss=round(epoch_avg, 6),
                epoch_time_s=round(epoch_time, 1),
                num_batches=num_batches,
            )


        result_model = None
        if self.config.save_dir:
            weights_path, num_saved = save_sparse_model(model, self.config.save_dir)
            self.logger.info(f'Sparse weights saved to {weights_path} ({num_saved} layers)')
            
            result_model = unwrap_distillation(model)
            self.logger.info('Model unwrapped: wrappers replaced with SparseSiLUFFN')
        else:
            result_model = unwrap_distillation(model)
            self.logger.warning('save_dir not set, model not saved to disk')

        return {
            'model': result_model,
            'final_avg_loss': epoch_avg,
            'total_epochs': self.config.epochs,
        }

    def log_result(self, result):
        if result is None:
            self.logger.warning('Distillation returned None')
            return
        
        self.logger.info(
            f'Distillation completed | '
            f'Final avg loss: {result["final_avg_loss"]:.6f} | '
            f'Epochs: {result["total_epochs"]}'
        )
