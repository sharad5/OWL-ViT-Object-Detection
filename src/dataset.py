import json
import os
from collections import Counter
from tqdm import tqdm
from collections import defaultdict

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch import nn
from transformers import OwlViTProcessor

TRAIN_ANNOTATIONS_FILE = "data/ego4d_queries_train.json"#"data/owlvit_train.json"
# VAL_ANNOTATIONS_FILE = "/scratch/hk3820/capstone/data/paco_annotations/paco_ego4d_v1_val.json"
IMAGES_PATH = "/scratch/hk3820/capstone/data/paco_frames/v1/paco_frames"

class Ego4dDataset(Dataset):
    def __init__(self, processor, annotations_file, num_pos_queries, num_neg_queries, is_train):
        self.images_dir = IMAGES_PATH
        self.processor = processor
        self.num_pos_queries = num_pos_queries
        self.num_neg_queries = num_neg_queries
        annotations_file = annotations_file

        with open(annotations_file) as f:
            if is_train:
                data = json.load(f)["annotations"][:-1000]
#                 data = np.random.choice(data, 10000)
            else:
                data = json.load(f)["annotations"][-1000:]

        self.data = data

    def load_image(self, idx: int) -> Image.Image:
        url = self.data[idx]["image_file_name"]
        path = os.path.join(self.images_dir, os.path.basename(url))
        image = Image.open(path).convert("RGB")
        return image, path

    def load_target(self, idx: int):
        all_pos_queries = self.data[idx]["pos_queries"]
        all_neg_queries = self.data[idx]["neg_queries"]
        
        if self.num_pos_queries > len(all_pos_queries):
            raise ValueError(f'Cannot select more positive queries than available. Image: {self.data[idx]["image_file_name"]}')
        if self.num_neg_queries > len(all_neg_queries):
            raise ValueError(f'Cannot select more negative queries than available. Image: {self.data[idx]["image_file_name"]}')
        
        text = [all_pos_queries[:self.num_pos_queries] + all_neg_queries[:self.num_neg_queries]]
        return text, self.data[idx]["bbox"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, path = self.load_image(idx)
        text, bbox = self.load_target(idx)
        
        # Target Labels
        num_queries = len(text[0])
        target_labels = nn.functional.one_hot(torch.arange(self.num_pos_queries).to(torch.int64), num_classes=num_queries)
        
        # Metadata generation
        w, h = image.size
        bbox = torch.tensor(bbox)
        bbox[[0,2]] /= w
        bbox[[1,3]] /= h
        metadata = {
            "width": w,
            "height": h,
            "impath": path,
            "num_pos_queries": self.num_pos_queries
        }
        
        inputs = self.processor(text=text, images=image, padding='max_length', max_length=16, truncation=True, return_tensors="pt")
        inputs['pixel_values'] = inputs['pixel_values'].squeeze(0)
        return inputs, target_labels, bbox, metadata

def get_dataloaders(cfg, processor):
    
    train_dataset = Ego4dDataset(
                        processor, 
                        annotations_file=cfg["train_data_file"], 
                        num_pos_queries=cfg["num_pos_queries"], 
                        num_neg_queries=cfg["num_neg_queries"],
                        is_train=True
                    )
    test_dataset = Ego4dDataset(
                        processor,
                        annotations_file=cfg["train_data_file"],
                        num_pos_queries=cfg["num_pos_queries"], 
                        num_neg_queries=cfg["num_neg_queries"],
                        is_train=False
                    )
    
    train_dataloader = DataLoader(train_dataset, batch_size=cfg["train_batch_size"], shuffle=True, num_workers=1)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg["train_batch_size"], shuffle=False, num_workers=1)
    
    return train_dataloader, test_dataloader
