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
from transformers import OwlViTProcessor

TRAIN_ANNOTATIONS_FILE = "data/ego4d_dummy_train.json"
# VAL_ANNOTATIONS_FILE = "/scratch/hk3820/capstone/data/paco_annotations/paco_ego4d_v1_val.json"
IMAGES_PATH = "/scratch/hk3820/capstone/data/paco_frames/v1/paco_frames"

class OwlDataset(Dataset):
    def __init__(self, processor, annotations_file):
        self.images_dir = IMAGES_PATH
        self.processor = processor
        annotations_file = annotations_file

        with open(annotations_file) as f:
            data = json.load(f)["annotations"]

        self.data = data

    def load_image(self, idx: int) -> Image.Image:
        url = self.data[idx]["image_file_name"]
        path = os.path.join(self.images_dir, os.path.basename(url))
        image = Image.open(path).convert("RGB")
        return image, path

    def load_target(self, idx: int):
        text = [self.data[idx]["pos_queries"] + self.data[idx]["neg_queries"]]
        return text, self.data[idx]["bbox"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, path = self.load_image(idx)
        text, bbox = self.load_target(idx)
        w, h = image.size
        metadata = {
            "width": w,
            "height": h,
            "impath": path,
        }
        
        inputs = self.processor(text=text, images=image, return_tensors="pt")
        inputs['pixel_values'] = inputs['pixel_values'].squeeze(0)
        return inputs, torch.tensor(bbox), metadata

def get_dataloaders(batch_size):
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    
    train_dataset = OwlDataset(processor, TRAIN_ANNOTATIONS_FILE)
    test_dataset = OwlDataset(processor, TRAIN_ANNOTATIONS_FILE)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    
    return train_dataloader, test_dataloader