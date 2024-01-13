import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

device = "cuda:0" if torch.cuda.is_available() else "cpu"
#dataset_file_name = "data/paco_annotations/paco_ego4d_v1_test_dev.json"
dataset_file_name = "/scratch/hk3820/capstone/data/paco_annotations/paco_ego4d_v1_test_dev.json"
image_root_dir = "/scratch/hk3820/capstone/data/paco_frames/v1/paco_frames"
model_checkpoint = "google/owlvit-base-patch32"
pt_model_path = "checkpoints/20231122_0651_model.pt"
img_batch_size = 4


def load_image_query_ids():
    with open(dataset_file_name) as f:
        dataset = json.load(f)
    all_queries = [d["query_string"] for d in dataset["queries"]]
    image_id_to_image = {d["id"]: d for d in dataset["images"]}
    image_id_to_image_file_name = {d["id"]: os.path.join(image_root_dir, d["file_name"]) for d in dataset["images"]}
    all_image_ids = list(image_id_to_image_file_name.keys())
    return all_queries, all_image_ids, image_id_to_image, image_id_to_image_file_name


def load_model(model_checkpoint):
    processor = OwlViTProcessor.from_pretrained(model_checkpoint) # Image Processor + Text Tokenizer
    model = OwlViTForObjectDetection.from_pretrained(model_checkpoint)
    model.load_state_dict(torch.load(pt_model_path)["model_state_dict"])
    model = model.to(device)
    return processor, model


def process_results_for_eval(results, image_ids):
    """ results = list(dicts) with scores, boxes, labels """
    processed_results = []
    for i in range(len(results)):
        res = results[i]
        pred = {}
        pred["image_id"] = image_ids[i]
        pred["bboxes"] = res["boxes"].cpu().numpy()
        pred["box_scores"] = res["scores"].cpu().numpy()
        pred["pred_classes"] = res["labels"].cpu().numpy()
        processed_results.append(pred)
    return processed_results


def get_batched_model_predictions(processor, model, all_queries, all_image_ids, image_id_to_image, image_id_to_image_file_name):
    predictions = []
    model.eval()
    
    for image_idx in tqdm(range(0, len(all_image_ids), img_batch_size)):
        image_ids = all_image_ids[image_idx: image_idx + img_batch_size]
        images = [Image.open(image_id_to_image_file_name[image_id]) for image_id in image_ids]
        inputs = processor(text=all_queries, images=images, return_tensors="pt", truncation=True)
        inputs = inputs.to(device)
        inputs["input_ids"] = inputs["input_ids"].repeat(img_batch_size,1)
        inputs["attention_mask"] = inputs["attention_mask"].repeat(img_batch_size,1)

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.Tensor([[image_id_to_image[image_id]["height"], image_id_to_image[image_id]["width"]] for image_id in image_ids])
        target_sizes = target_sizes.to(device)
        results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.05)
        
        prediction = process_results_for_eval(results, image_ids)
        predictions += prediction

        #del queries, inputs, outputs, target_sizes, results
        #torch.cuda.empty_cache()
        
    return predictions


def get_model_predictions(processor, model, all_queries, all_image_ids, image_id_to_image, image_id_to_image_file_name):
    predictions = []
    model.eval()

    for image_id in tqdm(all_image_ids):
        image = Image.open(image_id_to_image_file_name[image_id])
        inputs = processor(text=all_queries, images=image, return_tensors="pt", truncation=True) # Truncating!
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.Tensor([[image_id_to_image[image_id]["height"], image_id_to_image[image_id]["width"]]])
        target_sizes = target_sizes.to(device)
        results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)

        prediction = process_results_for_eval(results, [image_id])
        predictions += prediction
        
    return predictions


def export_results_pkl(predictions, filename):
    with open(f'data/inference/{filename}.pickle', 'wb') as f:
        pickle.dump(predictions, f, protocol=pickle.HIGHEST_PROTOCOL)
    

if __name__ == "__main__":
    all_queries, all_image_ids, image_id_to_image, image_id_to_image_file_name = load_image_query_ids()
    print(f"Loaded Dataset: {len(all_queries)} queries, {len(all_image_ids)} image_ids.")
    processor, model = load_model(model_checkpoint)
    print(f"Loaded Model: {model_checkpoint}")
    #predictions = get_model_predictions(processor, model, all_queries, all_image_ids, image_id_to_image, image_id_to_image_file_name)
    predictions = get_batched_model_predictions(processor, model, all_queries, all_image_ids, image_id_to_image, image_id_to_image_file_name)
    print("Exporting Results")
    model_name = model_checkpoint.split("/")[-1]
    #model_name += "_thr05"
    model_name += "_epoch3"
    export_results_pkl(predictions, model_name)
