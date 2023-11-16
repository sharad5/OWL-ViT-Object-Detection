import os
from datetime import datetime
from tqdm import tqdm
import torch
from torch import nn
from torchvision.ops import box_convert
import yaml
from transformers import OwlViTProcessor, OwlViTForObjectDetection

from src.dataset import get_dataloaders
from src.losses import ContrastiveDetectionLoss

def get_training_config():
    with open("config.yaml", "r") as stream:
        data = yaml.safe_load(stream)
        return data["training"]

def save_model_checkpoint(model, optimizer, epoch, loss, cfg):
    model_filename  = datetime.now().strftime("%Y%m%d_%H%M")+"_model.pt"
    path = os.path.join(cfg["model_checkpoint_path"], model_filename)
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, path)
    
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    training_cfg = get_training_config()
    
    owl_vit_checkpoint = training_cfg["owl_vit_checkpoint"] if training_cfg.get("owl_vit_checkpoint", None) \
                                                            else "google/owlvit-base-patch32"
    processor = OwlViTProcessor.from_pretrained(owl_vit_checkpoint) # Image Processor + Text Tokenizer
    train_dataloader, test_dataloader = get_dataloaders(
                                            batch_size=training_cfg["train_batch_size"],
                                            processor=processor
                                        )
    
    
    model = OwlViTForObjectDetection.from_pretrained(owl_vit_checkpoint)
    model = model.to(device)
    
    criterion = ContrastiveDetectionLoss()
    
    optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=float(training_cfg["learning_rate"]),
                    weight_decay=training_cfg["weight_decay"],
                )
    
    num_epochs = training_cfg["n_epochs"]
    num_training_steps = num_epochs * len(train_dataloader)
    progress_bar = tqdm(range(num_training_steps))
    
    model.train()
    for epoch in range(training_cfg["n_epochs"]):
        for i, (inputs, target_labels, boxes, metadata) in enumerate(train_dataloader):
            optimizer.zero_grad()

            inputs['input_ids'] = inputs['input_ids'].view(-1,16)
            inputs['attention_mask'] = inputs['attention_mask'].view(-1,16)

            inputs = inputs.to(device)

            outputs = model(**inputs)


            logits = outputs["logits"]
            pred_boxes = outputs["pred_boxes"]

            batch_size = boxes.shape[0]

            target_labels = target_labels.to(device)
            boxes = boxes.to(device)

            loss = criterion(logits, pred_boxes, boxes, target_labels, metadata)
            loss.backward()
            optimizer.step()
            progress_bar.update(1)
            progress_bar.set_description(f"Loss: {loss.item():.3f}")
            
        save_model_checkpoint(model, optimizer, epoch, loss, training_cfg)