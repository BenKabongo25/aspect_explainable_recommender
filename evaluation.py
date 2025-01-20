# Ben Kabongo
# January 2025

import evaluate
import numpy as np
import torch
import torch.nn.functional as F

from typing import Any, Dict, List


RMSE = "rmse"
MAE = "mae"
RATING_METRICS = [RMSE, MAE]

PRECISION = "precision"
RECALL = "recall"
F1 = "f1"
NDCG = "ndcg"
MAP = "map"
RANKING_METRICS = [PRECISION, RECALL, F1, NDCG, MAP]

BLEU = "bleu"
METEOR = "meteor"
ROUGE = "rouge"
BERTSCORE = "bertscore"
REVIEW_METRICS = [BLEU, METEOR, ROUGE, BERTSCORE]



def rating_evaluation(
        config: Any, predictions: List[float], references: List[float], 
        users: List[float]=None, metrics: List[str]=[RMSE, MAE],
    ) -> Dict[str, float]:
    
    results = {}
    
    actual_ratings = torch.tensor(references, dtype=torch.float32).to(config.device)
    predictions_tensor = torch.tensor(predictions, dtype=torch.float32).to(config.device)

    if RMSE in metrics:
        rmse = torch.sqrt(F.mse_loss(predictions_tensor, actual_ratings))
        results[RMSE] = rmse.item()
        
    if MAE in metrics:
        mae = F.l1_loss(predictions_tensor, actual_ratings)
        results[MAE] = mae.item()
    mae = F.l1_loss(predictions_tensor, actual_ratings)

    if not set(metrics).intersection(RANKING_METRICS):
        return results

    threshold = torch.tensor(config.threshold_rating).to(config.device)
    actual_binary = (actual_ratings >= threshold).float()
    predicted_binary = (predictions_tensor >= threshold).float()
    true_positives = (predicted_binary * actual_binary).sum()

    precision = true_positives / predicted_binary.sum() if predicted_binary.sum() > 0 else torch.tensor(1.0).to(config.device)
    recall = true_positives / actual_binary.sum() if actual_binary.sum() > 0 else torch.tensor(1.0).to(config.device)
    
    if PRECISION in metrics:
        results[PRECISION] = precision.item()
    
    if RECALL in metrics:
        results[RECALL] = recall.item()
    
    if F1 in metrics:
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else torch.tensor(0.0).to(config.device)
        results[F1] = f1.item()

    if not set(metrics).intersection([NDCG, MAP]):
        return results
    
    users = list(set(users))
    
    if NDCG in metrics:
        ndcg_scores = []

    if MAP in metrics:
        average_precisions = []
    
    for user in users:
        user_predictions = [pred for idx, pred in enumerate(predictions) if users[idx] == user]
        user_references = [ref for idx, ref in enumerate(references) if users[idx] == user]

        user_predictions_tensor = torch.tensor(user_predictions, dtype=torch.float32).to(config.device)
        
        sorted_indices = torch.configort(user_predictions_tensor, descending=True)
        relevances = torch.tensor(user_references, dtype=torch.float32).to(config.device)[sorted_indices]

        if NDCG in metrics:
            ndcg = ndcg_at_k(relevances, config.k, config.device)
            ndcg_scores.append(ndcg.item())

        if MAP in metrics:
            ap = average_precision(relevances, config.threshold_rating, config.device)
            average_precisions.append(ap.item())

    if NDCG in metrics:
        ndcg = torch.tensor(ndcg_scores).mean().item()
        results[NDCG] = ndcg

    if MAP in metrics:
        map = torch.tensor(average_precisions).mean().item()
        results[MAP] = map

    return results


def review_evaluation(
        config, predictions: List[str], references: List[str],
        metrics: List[str]=[BLEU, METEOR, ROUGE, BERTSCORE]
    ) -> Dict[str, Any]:

    results = {}
    references_list = [[ref] for ref in references]

    if BLEU in metrics:
        bleu_metric = evaluate.load("bleu")
        bleu_results = bleu_metric.compute(predictions=predictions, references=references_list)
        results[BLEU] = bleu_results["bleu"]

    if METEOR in metrics:
        meteor_metric = evaluate.load("meteor")
        meteor_results = meteor_metric.compute(predictions=predictions, references=references)
        results[METEOR] = meteor_results["meteor"]

    if ROUGE in metrics:
        rouge_metric = evaluate.load("rouge")
        rouge_results = rouge_metric.compute(predictions=predictions, references=references)
        results[ROUGE + ".1"] = rouge_results["rouge1"]
        results[ROUGE + ".2"] = rouge_results["rouge2"]
        results[ROUGE + ".L"] = rouge_results["rougeL"]
        results[ROUGE + ".Lsum"] = rouge_results["rougeLsum"]

    if BERTSCORE in metrics:
        bertscore_metric = evaluate.load("bertscore")
        bertscore_results = bertscore_metric.compute(
            predictions=predictions, references=references, lang=config.lang, device=config.device
        )
        results[BERTSCORE + ".precision"] = np.mean(bertscore_results["precision"])
        results[BERTSCORE + ".recall"] = np.mean(bertscore_results["recall"])
        results[BERTSCORE + ".f1"] = np.mean(bertscore_results["f1"])

    return results


def dcg_at_k(relevances, k, device):
    relevances = relevances[:k]
    positions = torch.arange(1, len(relevances) + 1, dtype=torch.float32).to(device)
    return torch.sum(relevances / torch.log2(positions + 1))


def ndcg_at_k(relevances, k, device):
    dcg = dcg_at_k(relevances, k, device)
    ideal_relevances = torch.sort(relevances, descending=True).values
    idcg = dcg_at_k(ideal_relevances, k, device)
    return dcg / idcg if idcg > 0 else torch.tensor(0.0).to(device)


def average_precision(relevances, threshold_rating, device):
    precisions = []
    num_relevant = 0
    for k in range(1, len(relevances) + 1):
        if relevances[k - 1] >= threshold_rating:
            num_relevant += 1
            precisions.append(num_relevant / k)

    if num_relevant == 0:
        return torch.tensor(0.0).to(device)
    
    return torch.tensor(precisions).mean().to(device)
