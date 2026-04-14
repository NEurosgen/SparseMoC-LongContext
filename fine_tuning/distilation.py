from fine_tuning.abstract_pipeline import Pipeline
import torch
from transformers import AutoModelForCausalLM
from torch import nn
import torch.nn.functional as F
import os
import time
import math

from models.moc_ffn_triton.SparseSiLUFFN import SparseSiLUFFN
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


def load_sparse_model(model_path, sparse_weights_path, device='cuda'):
    """Загрузка модели с sparse FFN весами."""
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype="auto", device_map=device,
        attn_implementation="sdpa"
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
    """
    Wrapper для layer-wise дистилляции.
    """
    def __init__(self, old_mlp, new_mlp, cosine_weight=0.5):
        super().__init__()
        self.old_mlp = old_mlp
        self.new_mlp = new_mlp
        self.cosine_weight = cosine_weight
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
            mse = F.mse_loss(pred_out, target_out)
            cos_sim = F.cosine_similarity(
                pred_out.reshape(-1, pred_out.shape[-1]),
                target_out.reshape(-1, target_out.shape[-1]),
                dim=-1
            ).mean()
            self.distill_loss = mse + self.cosine_weight * (1.0 - cos_sim)
        
        return target_out


def prepare_model_dist(model, top_k, cosine_weight=0.5):
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
            
        wrapper = FFNDistillationWrapper(old_mlp, new_mlp, cosine_weight=cosine_weight)
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
    Двухфазная дистилляция FFN-слоёв:
    
    Phase 1: MSE + cosine similarity на FFN выходе.
        Быстрая, выравнивает веса sparse FFN к dense.
    
    Phase 2: KL-divergence на logits всей модели.
        Сохраняет downstream quality, учит модель end-to-end.

    """

    def __init__(self, config):
        super().__init__(config=config)

    def _run_phase1(self, model, data, optimizer, scheduler, profiler=None):
        """Phase 1: Layer-wise MSE + Cosine similarity."""
        num_layers = len(model.model.layers)
        epochs = getattr(self.config, 'epochs_phase1', 2)
        total_batches = len(data)

        self.logger.info(f'=== PHASE 1: Layer-wise distillation ({epochs} epochs) ===')

        for epoch in range(epochs):
            epoch_start = time.time()
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(data):
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(model.device)
                
                with torch.no_grad():
                    _ = model(input_ids)
                
                layer_losses = []
                total_loss = torch.tensor(0.0, device=model.device)
                for layer in model.model.layers:
                    l = layer.mlp.distill_loss
                    layer_losses.append(l.item())
                    total_loss = total_loss + l
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                if scheduler:
                    scheduler.step()
                
                batch_loss = total_loss.item()
                epoch_loss += batch_loss
                num_batches += 1

                if batch_idx % 10 == 0:
                    avg_layer = batch_loss / num_layers
                    current_lr = optimizer.param_groups[0]['lr']
                    self.logger.info(
                        f'P1 Epoch {epoch}/{epochs-1} | '
                        f'Batch {batch_idx}/{total_batches} | '
                        f'Loss: {batch_loss:.6f} | '
                        f'Avg/layer: {avg_layer:.6f} | '
                        f'LR: {current_lr:.2e}'
                    )
                    self.log_metric(
                        stage='distill_phase1',
                        epoch=epoch, batch=batch_idx,
                        total_loss=round(batch_loss, 6),
                        avg_layer_loss=round(avg_layer, 6),
                        lr=current_lr,
                    )

                if profiler:
                    profiler.step()

            epoch_time = time.time() - epoch_start
            epoch_avg = epoch_loss / max(num_batches, 1)
            self.logger.info(
                f'--- P1 Epoch {epoch}: avg_loss={epoch_avg:.6f}, '
                f'time={epoch_time:.1f}s ---'
            )
            self.log_metric(
                stage='distill_phase1_epoch',
                epoch=epoch, avg_loss=round(epoch_avg, 6),
                epoch_time_s=round(epoch_time, 1),
            )

        return epoch_avg

    def _run_phase2(self, model, base_model_path, data, profiler=None):
        """
        Phase 2: End-to-end KL-divergence на logits.
        
        Загружает оригинальную  модель, прогоняет оба,
        считает KL(sparse_logits || dense_logits).
        """
        epochs = getattr(self.config, 'epochs_phase2', 1)
        if epochs <= 0:
            self.logger.info('Phase 2 skipped (epochs_phase2=0)')
            return 0.0

        kl_temperature = getattr(self.config, 'kl_temperature', 2.0)
        lr_phase2 = getattr(self.config, 'lr_phase2', self.config.lr * 0.5)

        self.logger.info(f'=== PHASE 2: End-to-end KL distillation ({epochs} epochs) ===')
        self.logger.info(f'KL temperature: {kl_temperature}, LR: {lr_phase2}')

        unwrap_distillation(model)
        self.logger.info('Wrappers unwrapped for end-to-end forward pass')

        self.logger.info(f'Loading teacher model from {base_model_path}...')
        teacher = AutoModelForCausalLM.from_pretrained(
            base_model_path, torch_dtype="auto", device_map=model.device
        )
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        self.logger.info('Teacher loaded')

        sparse_params = []
        for layer in model.model.layers:
            if isinstance(layer.mlp, SparseSiLUFFN):
                sparse_params.extend(layer.mlp.parameters())

        for p in sparse_params:
            p.requires_grad = True

        for p in model.parameters():
            if not any(p is sp for sp in sparse_params):
                p.requires_grad = False

        optimizer = AdamW(sparse_params, lr=lr_phase2)
        total_batches = len(data)
        total_steps = total_batches * epochs
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

        for epoch in range(epochs):
            epoch_start = time.time()
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(data):
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(model.device)

                with torch.no_grad():
                    teacher_outputs = teacher(input_ids=input_ids)
                    teacher_logits = teacher_outputs.logits

                student_outputs = model(input_ids=input_ids)
                student_logits = student_outputs.logits

                T = kl_temperature
                loss = F.kl_div(
                    F.log_softmax(student_logits / T, dim=-1),
                    F.softmax(teacher_logits / T, dim=-1),
                    reduction='batchmean'
                ) * (T * T)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(sparse_params, 1.0)
                optimizer.step()
                scheduler.step()

                batch_loss = loss.item()
                epoch_loss += batch_loss
                num_batches += 1

                if batch_idx % 10 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    self.logger.info(
                        f'P2 Epoch {epoch}/{epochs-1} | '
                        f'Batch {batch_idx}/{total_batches} | '
                        f'KL Loss: {batch_loss:.6f} | '
                        f'LR: {current_lr:.2e}'
                    )
                    self.log_metric(
                        stage='distill_phase2',
                        epoch=epoch, batch=batch_idx,
                        kl_loss=round(batch_loss, 6),
                        lr=current_lr,
                    )

                if profiler:
                    profiler.step()

            epoch_time = time.time() - epoch_start
            epoch_avg = epoch_loss / max(num_batches, 1)
            self.logger.info(
                f'--- P2 Epoch {epoch}: avg_kl_loss={epoch_avg:.6f}, '
                f'time={epoch_time:.1f}s ---'
            )

        del teacher
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        self.logger.info('Teacher model unloaded')

        return epoch_avg

    def process(self, model, data, profiler=None):
        """
        Двухфазная дистилляция.
        data — dataloader.
        """
        num_layers = len(model.model.layers)
        top_k = self.config.top_k
        epochs_p1 = getattr(self.config, 'epochs_phase1', 2)
        epochs_p2 = getattr(self.config, 'epochs_phase2', 1)
        cosine_weight = getattr(self.config, 'cosine_weight', 0.5)

        self.logger.info(f'Model layers: {num_layers}, top_k: {top_k}')
        self.logger.info(f'Phase 1: {epochs_p1} epochs (MSE+Cosine, cosine_w={cosine_weight})')
        self.logger.info(f'Phase 2: {epochs_p2} epochs (KL on logits)')
        self.logger.info(f'LR: {self.config.lr}')


        prepare_model_dist(model, top_k, cosine_weight=cosine_weight)
        self.logger.info('Model wrapped with FFNDistillationWrapper')

        trainable_params = []
        for layer in model.model.layers:
            trainable_params.extend(list(layer.mlp.new_mlp.parameters()))
        
        total_trainable = sum(p.numel() for p in trainable_params)
        self.logger.info(f'Trainable parameters: {total_trainable:,}')
        
        optimizer = AdamW(trainable_params, lr=self.config.lr)
        total_steps_p1 = len(data) * epochs_p1
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps_p1)
        model.eval()

        p1_loss = self._run_phase1(model, data, optimizer, scheduler, profiler)

        if self.config.save_dir:
            weights_path, num_saved = save_sparse_model(model, self.config.save_dir)
            self.logger.info(f'Phase 1 weights saved to {weights_path} ({num_saved} layers)')

        base_model_path = getattr(self.config, 'base_model_path', None)
        if epochs_p2 > 0 and base_model_path:
            p2_loss = self._run_phase2(model, base_model_path, data, profiler)
        else:
            if epochs_p2 > 0:
                self.logger.warning('Phase 2 skipped: base_model_path not set in config')
            p2_loss = 0.0
            unwrap_distillation(model)

        if self.config.save_dir:
            weights_path, num_saved = save_sparse_model(model, self.config.save_dir)
            self.logger.info(f'Final weights saved to {weights_path} ({num_saved} layers)')

        self.logger.info('Distillation complete')

        return {
            'model': model,
            'phase1_loss': p1_loss,
            'phase2_loss': p2_loss,
            'final_avg_loss': p2_loss if epochs_p2 > 0 else p1_loss,
            'total_epochs': epochs_p1 + epochs_p2,
        }

    def log_result(self, result):
        if result is None:
            self.logger.warning('Distillation returned None')
            return
        
        self.logger.info(
            f'Distillation completed | '
            f'P1 loss: {result.get("phase1_loss", "N/A")} | '
            f'P2 loss: {result.get("phase2_loss", "N/A")} | '
            f'Epochs: {result["total_epochs"]}'
        )
