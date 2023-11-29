import os
from datetime import datetime
from tqdm import tqdm
from collections import OrderedDict
import torch
from torch import nn
from torchvision.ops import box_convert
import yaml
from transformers import OwlViTProcessor, OwlViTForObjectDetection

from src.dataset import get_dataloaders
from src.losses import ContrastiveDetectionLoss
import wandb
import logging


def get_training_config():
    with open("config.yaml", "r") as stream:
        data = yaml.safe_load(stream)
        return data["training"]

def save_model_checkpoint(model, optimizer, epoch, loss, cfg, wandb_identifier):
    model_filename  = f"{wandb_identifier}_epoch-{epoch}.pt"#datetime.now().strftime("%Y%m%d_%H%M")+"_model.pt"
    path = os.path.join(cfg["model_checkpoint_path"], model_filename)
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, path)

def acquire_device(cfg):
    num_gpus = 0
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg["gpu"]
        device = torch.device('cuda')
        num_gpus = torch.cuda.device_count()
        print(f'Using {num_gpus} GPUs: {cfg["gpu"]}')
    else:
        device = torch.device('cpu')
        print('Use CPU')
    return device, num_gpus

def build_model(cfg, device, num_gpus):
    owl_vit_checkpoint = cfg["owl_vit_checkpoint"] if cfg.get("owl_vit_checkpoint", None) \
                                                   else "google/owlvit-base-patch32"
    model = OwlViTForObjectDetection.from_pretrained(owl_vit_checkpoint)
    
    # Load from saved checkpoint
    if cfg["use_model_checkpoint"]:
        model_path = os.path.join(cfg["model_checkpoint_path"], cfg["model_checkpoint_file"])
        state_dict = torch.load(model_path)["model_state_dict"]
        if "module" in list(state_dict.keys())[0]:
            state_dict = OrderedDict({".".join(k.split(".")[1:]): v for k,v in state_dict.items()})
        model.load_state_dict(state_dict)
    
    # DDP
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    return model
    
if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    training_cfg = get_training_config()
    device, num_gpus = acquire_device(training_cfg)
    owl_vit_checkpoint = training_cfg["owl_vit_checkpoint"] if training_cfg.get("owl_vit_checkpoint", None) \
                                                            else "google/owlvit-base-patch32"
    processor = OwlViTProcessor.from_pretrained(owl_vit_checkpoint) # Image Processor + Text Tokenizer
    train_dataloader, test_dataloader = get_dataloaders(
                                            cfg=training_cfg,
                                            processor=processor
                                        )
    
    
    model = build_model(training_cfg, device, num_gpus)
    
    criterion = ContrastiveDetectionLoss(focal_loss_coef = training_cfg["focal_loss_coef"],
                                         bbox_loss_coef = training_cfg["bbox_loss_coef"],
                                         giou_loss_coef = training_cfg["giou_loss_coef"],
                                         non_overlap_loss_coef = training_cfg["non_overlap_loss_coef"],
                                         non_overlap_giou_thresh = training_cfg["non_overlap_giou_thresh"]
                                        )
    
    
    optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=float(training_cfg["learning_rate"]),
                    weight_decay=training_cfg["weight_decay"],
                )
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8, verbose=True)
    
    num_epochs = training_cfg["n_epochs"]
    num_batches = len(train_dataloader)
    num_training_steps = num_epochs * num_batches
    progress_bar = tqdm(range(num_training_steps))
    
    wandb_run = wandb.init(project="owl-vit", entity="a-is-all-we-need", config=training_cfg)#, mode="disabled")
    wandb_identifier = wandb_run.name
    #wandb.watch(model, log_freq=50)
    
    model.train()
    step = 0
    for epoch in range(training_cfg["n_epochs"]):
        total_loss, total_focal_loss, total_bbox_loss, total_giou_loss, total_non_overlap_loss = 0,0,0,0,0
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
            try:
                loss, focal_loss, bbox_loss, giou_loss, non_overlap_loss = criterion(logits, pred_boxes, boxes, target_labels, metadata, step)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                total_focal_loss += focal_loss.item()
                total_bbox_loss += bbox_loss.item()
                total_giou_loss += giou_loss.item()
                total_non_overlap_loss += non_overlap_loss.item()

                progress_bar.update(1)
                progress_bar.set_description(f"Loss: {loss.item():.3f}")
                wandb.log({"train_loss": loss, "focal_loss": focal_loss, "bbox_loss": bbox_loss, "giou_loss": giou_loss, "non_overlap_loss":non_overlap_loss})
                step+=1
            except Exception as e:
                print(f"Step: {step}", type(e))
                logger.error(str(e), exc_info=True)
                step+=1
                progress_bar.update(1)
                continue
        
        scheduler.step()
        mean_total_loss = total_loss/num_batches
        mean_focal_loss = total_focal_loss/num_batches
        mean_bbox_loss = bbox_loss/num_batches
        mean_giou_loss = giou_loss/num_batches
        mean_non_overlap_loss = non_overlap_loss/num_batches
        print(f"Loss: {mean_total_loss:.3f}, Focal Loss: {mean_focal_loss:.3f}, BBox Loss: {mean_bbox_loss:.3f}, GIOU Loss: {mean_giou_loss:.3f}, Non-Overlap Loss: {mean_non_overlap_loss:.3f}")
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
                try:
                    val_loss, _, _, _, _ = criterion(logits, pred_boxes, boxes, target_labels, metadata, step)
#                 loss.backward()
#                 optimizer.step()
                    total_val_loss += val_loss.item()
                except Exception as e:
                    print(f"Validation at Epoch : {epoch}", type(e))
                    continue
                torch.cuda.empty_cache()
        
        
        wandb.log({"epoch_train_loss": mean_total_loss, "epoch_train_focal_loss": mean_focal_loss, "epoch_train_bbox_loss": mean_bbox_loss, "epoch_train_giou_loss": mean_giou_loss, "epoch_train_non_overlap_loss": mean_non_overlap_loss, "epoch_val_loss": total_val_loss/len(test_dataloader)})
            
        save_model_checkpoint(model, optimizer, epoch, mean_total_loss, training_cfg, wandb_identifier)
