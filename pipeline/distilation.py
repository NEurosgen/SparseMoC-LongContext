from pipeline.abstract_pipe import Pipeline
import torch
from transformers import AutoModelForCausalLM
from torch import nn
import torch.nn.functional as F
import os
import sys

# Добавляем корень проекта в sys.path для импорта моделей
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.moc_ff_triton import SparseSiLUFFN
from torch.optim import AdamW
def load_sparse_model(model_path, sparse_weights_path, device='cuda'):

    
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
    
    print(f"Loaded sparse FFN weights for {len(sparse_weights)} layers")
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
    print(f"Sparse FFN weights saved to {weights_path}")
    print(f"  Layers saved: {len(sparse_weights)}")
    if sparse_weights:
        first = next(iter(sparse_weights.values()))
        print(f"  top_k={first['top_k']}, d_model={first['d_model']}, d_ffn={first['d_ffn']}")



class Distilation(Pipeline):
    '''
    Класс производит дистялцую полбзовательски слоев
    Интерфейс взаимодействия с моделью должен оперделять пользователь
    Но пока и так норм
    '''
    def __init__(self, config):
        super().__init__(config=config)

    def process(self, model, data):
        '''
        В качестве data  призодит dataloader
        '''
        prepare_model_dist(model , self.config.top_k)

        trainable_params = []
        for layer in model.model.layers:
            trainable_params.extend(list(layer.mlp.new_mlp.parameters()))
        
        optimizer = AdamW(trainable_params, lr=self.config.lr)
        
        model.eval()
        
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_idx, batch in enumerate(data):
                optimizer.zero_grad()
                
                input_ids = batch["input_ids"].to(model.device) # Надо испраивть labels и всякое такое .Неправильно
                
                with torch.no_grad():
                    _ = model(input_ids)
                loss = torch.tensor(0.0, device=model.device)
                for layer in model.model.layers:
                    loss = loss + layer.mlp.distill_loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    avg_layer_loss = loss.item() / len(model.model.layers)
                    print(f"Epoch {epoch} | Batch {batch_idx} | "
                        f"Total Loss: {loss.item():.6f} | "
                        f"Avg Layer Loss: {avg_layer_loss:.6f}")
        

        if self.config.save_dir:
            save_sparse_model(model, self.config.save_dir)
            result = unwrap_distillation(model)
            print(f"Model unwrapped: wrappers replaced with SparseSiLUFFN")
        return result
            
