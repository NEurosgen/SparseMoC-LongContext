from pipeline.abstract_pipe import Pipeline

from  torch.optim import AdamW
import torch
import os
def unfreeze_ffn_layer(model):
    for  i in range(len(model.model.layers)):
        for param in model.model.layers[i].mlp.parameters():
            param.requires_grad = True

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
class EndToEndFT(Pipeline):
    def __init__(self, config):
        super().__init__(config)

    def process(self, model, data, profiler = None):
        freeze_model(model)
        unfreeze_ffn_layer(model)
        traineble_parametrs = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(params=traineble_parametrs, lr= self.config.lr)

        for epoch in range(self.config.epochs):
            total_loss = 0.0
            for batch_idx, batch in enumerate(data):
                optimizer.zero_grad()
                
                input_ids = batch["input_ids"].to(model.device)
                labels = batch["labels"].to(model.device)
                outputs = model(input_ids = input_ids, labels = labels)
                loss = outputs.loss

                
                loss.backward()
                optimizer.step()
                
                if profiler:
                    profiler.step()

                if batch_idx % 10 == 0: 
                    print(f"Epoch {epoch} | Batch {batch_idx} | "
                        f"Total Loss: {loss.item():.6f} | ")
                if profiler and batch_idx > 15:
                    return
    def log_result(self, result):
        log_dir = getattr(self.config, 'log_dir', './logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "finetune_log.txt")
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write("End-to-End Fine-Tuning epoch/process completed.\n")
                        
        


            

