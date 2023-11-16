import json
import pickle
from paco.evaluation import paco_query_evaluation

PACO_TEST_PATH = "paco_data/annotations/paco_ego4d_v1_test_dev.json"
MODEL_PREDS_PATH = "paco_data/inference/owlvit-base-patch32.pickle"

def calculate_ar():
    with open(PACO_TEST_PATH) as f:
        dataset = json.load(f)

    with open(MODEL_PREDS_PATH, "rb") as f:
        model_preds = pickle.load(f)

    model_evaluator = paco_query_evaluation.PACOQueryPredictionEvaluator(dataset, model_preds)
    img_id_to_query_pos_scores = model_evaluator.evaluation_loop(deduplicate_boxes=False, print_results=True)
    results_model = model_evaluator.get_results()
    return results_model # 'AR@1_ALL_ALL'

if __name__ == "__main__":
    calculate_ar()