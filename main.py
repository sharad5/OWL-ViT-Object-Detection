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
import wandb

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
                                            cfg=training_cfg,
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
    num_batches = len(train_dataloader)
    num_training_steps = num_epochs * num_batches
    progress_bar = tqdm(range(num_training_steps))
    
    wandb.init(project="owl-vit", config=training_cfg)
    #wandb.watch(model, log_freq=20)
    
    model.train()
    step = 0
    for epoch in range(training_cfg["n_epochs"]):
        total_loss, total_focal_loss, total_bbox_loss, total_giou_loss = 0,0,0,0
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

            loss, focal_loss, bbox_loss, giou_loss = criterion(logits, pred_boxes, boxes, target_labels, metadata, step)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_focal_loss += focal_loss.item()
            total_bbox_loss += bbox_loss.item()
            total_giou_loss += giou_loss.item()
            
            progress_bar.update(1)
            progress_bar.set_description(f"Loss: {loss.item():.3f}")
            wandb.log({"train_loss": loss, "focal_loss": focal_loss, "bbox_loss": bbox_loss, "giou_loss": giou_loss})
            step+=1
        
        mean_total_loss = total_loss/num_batches
        mean_focal_loss = total_focal_loss/num_batches
        mean_bbox_loss = bbox_loss/num_batches
        mean_giou_loss = giou_loss/num_batches
        print(f"Loss: {mean_total_loss:.3f}, Focal Loss: {mean_focal_loss:.3f}, BBox Loss: {mean_bbox_loss:.3f}, GIOU Loss: {mean_giou_loss:.3f}")
        total_val_loss = 0
        model.eval()
        with torch.no_grad():
            
            for i, (inputs, target_labels, boxes, metadata) in enumerate(test_dataloader):
                inputs['input_ids'] = inputs['input_ids'].view(-1,16)
                inputs['attention_mask'] = inputs['attention_mask'].view(-1,16)

                inputs = inputs.to(device)

                outputs = model(**inputs)
                
                logits = outputs["logits"]
                pred_boxes = outputs["pred_boxes"]

                batch_size = boxes.shape[0]

                target_labels = target_labels.to(device)
                boxes = boxes.to(device)

                val_loss, _, _, _ = criterion(logits, pred_boxes, boxes, target_labels, metadata, step)
#                 loss.backward()
#                 optimizer.step()
                total_val_loss += val_loss.item()
                torch.cuda.empty_cache()
        
        
        wandb.log({"epoch_train_loss": mean_total_loss, "epoch_train_focal_loss": mean_focal_loss, "epoch_train_bbox_loss": mean_bbox_loss, "epoch_train_giou_loss": mean_giou_loss, "epoch_val_loss": total_val_loss/len(test_dataloader)})
            
        save_model_checkpoint(model, optimizer, epoch, mean_total_loss, training_cfg)
