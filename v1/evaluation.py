# Ben Kabongo
# November 2024

from utils import *


def rating_evaluation_pytorch(config: Any, 
                              predictions: List[float], references: List[float], 
                              users: List[float]=None) -> Dict[str, float]:
    
    results = {}
    
    actual_ratings = torch.tensor(references, dtype=torch.float32).to(config.device)
    predictions_tensor = torch.tensor(predictions, dtype=torch.float32).to(config.device)

    rmse = torch.sqrt(F.mse_loss(predictions_tensor, actual_ratings))
    mae = F.l1_loss(predictions_tensor, actual_ratings)
    
    results.update({'rmse': rmse.item(), 'mae': mae.item()})

    threshold = torch.tensor(config.threshold_rating).to(config.device)
    actual_binary = (actual_ratings >= threshold).float()
    predicted_binary = (predictions_tensor >= threshold).float()

    true_positives = (predicted_binary * actual_binary).sum()
    precision = true_positives / predicted_binary.sum() if predicted_binary.sum() > 0 else torch.tensor(1.0).to(config.device)
    recall = true_positives / actual_binary.sum() if actual_binary.sum() > 0 else torch.tensor(1.0).to(config.device)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else torch.tensor(0.0).to(config.device)

    results.update({'precision': precision.item(), 'recall': recall.item(), 'f1': f1.item()})

    if not config.ranking_metrics_flag and users is None:
        return results

    users = set(users)
    
    ndcg_scores = []
    average_precisions = []
    #reciprocal_ranks = []
    
    for user in users:
        user_predictions = [pred for idx, pred in enumerate(predictions) if users[idx] == user]
        user_references = [ref for idx, ref in enumerate(references) if users[idx] == user]

        user_predictions_tensor = torch.tensor(user_predictions, dtype=torch.float32).to(config.device)
        
        sorted_indices = torch.configort(user_predictions_tensor, descending=True)
        relevances = torch.tensor(user_references, dtype=torch.float32).to(config.device)[sorted_indices]

        ndcg = ndcg_at_k_pytorch(relevances, config.k, config.device)
        ndcg_scores.append(ndcg.item())

        ap = average_precision_pytorch(relevances, config.threshold_rating, config.device)
        average_precisions.append(ap.item())

        #for rank, relevance in enumerate(relevances, start=1):
        #    if relevance >= config.threshold_rating:
        #        reciprocal_ranks.append(1 / rank)
        #        break
        #else:
        #   reciprocal_ranks.append(0)

    ndcg = torch.tensor(ndcg_scores).mean().item()
    map = torch.tensor(average_precisions).mean().item()
    #mrr = torch.tensor(reciprocal_ranks).mean().item()

    results.update({'ndcg': ndcg, 'map': map})

    return results


def rating_evaluation(config: Any, 
                      predictions: List[float], references: List[float], 
                      users: List[float]=None) -> Dict[str, float]:
    
    results = {}

    actual_ratings = references
    
    rmse = np.sqrt(mean_squared_error(actual_ratings, predictions))
    mae = mean_absolute_error(actual_ratings, predictions)

    results.update({'rmse': rmse, 'mae': mae})

    binarizer = Binarizer(threshold=config.threshold_rating)
    actual_binary = binarizer.fit_transform(actual_ratings.reshape(-1, 1)).flatten()
    predicted_binary = binarizer.transform(np.array(predictions).reshape(-1, 1)).flatten()

    precision = precision_score(actual_binary, predicted_binary, zero_division=1)
    recall = recall_score(actual_binary, predicted_binary, zero_division=1)
    f1 = f1_score(actual_binary, predicted_binary, zero_division=1)

    results.update({'precision': precision, 'recall': recall, 'f1': f1})

    if not config.ranking_metrics_flag and users is None:
        return results

    users = set(users)
    
    ndcg_scores = []
    average_precisions = []
    #reciprocal_ranks = []
    
    for user in users:
        user_predictions = [pred for idx, pred in enumerate(predictions) if users[idx] == user]
        user_references = [ref for idx, ref in enumerate(references) if users[idx] == user]
        
        user_df.loc[:, 'predicted'] = user_predictions
        user_df = user_df.sort_values(by='predicted', ascending=False)
        
        relevances = user_references
        
        ndcg = ndcg_at_k(relevances, config.k)
        ndcg_scores.append(ndcg)
        
        ap = average_precision(relevances, config.threshold_rating)
        average_precisions.append(ap)

        #for rank, relevance in enumerate(relevances, start=1):
        #    if relevance >= config.threshold_rating:
        #        reciprocal_ranks.append(1 / rank)
        #        break
        #else:
        #    reciprocal_ranks.append(0)
    
    ndcg = np.mean(ndcg_scores)
    map = np.mean(average_precisions)
    #mrr = np.mean(reciprocal_ranks)

    results.update({'ndcg': ndcg, 'map': map})

    return results


def review_evaluation(config, predictions: List[str], references: List[str]) -> Dict[str, Any]:
    references_list = [[ref] for ref in references]

    bleu_metric = evaluate.load("bleu")
    bleu_results = bleu_metric.compute(predictions=predictions, references=references_list)
    bleu_results["precision"] = np.mean(bleu_results["precisions"])

    bertscore_metric = evaluate.load("bertscore")
    bertscore_results = bertscore_metric.compute(
        predictions=predictions, references=references, lang=config.lang
    )
    bertscore_results["precision"] = np.mean(bertscore_results["precision"])
    bertscore_results["recall"] = np.mean(bertscore_results["recall"])
    bertscore_results["f1"] = np.mean(bertscore_results["f1"])

    meteor_metric = evaluate.load("meteor")
    meteor_results = meteor_metric.compute(predictions=predictions, references=references)

    rouge_metric = evaluate.load("rouge")
    rouge_results = rouge_metric.compute(predictions=predictions, references=references)

    return {
        "n_examples": len(predictions),
        "meteor": float(meteor_results["meteor"]),
        "bleu": float(bleu_results["bleu"]),
        "rouge1": float(rouge_results["rouge1"]),
        "rouge2": float(rouge_results["rouge2"]),
        "rougeL": float(rouge_results["rougeL"]),
        "rougeLsum": float(rouge_results["rougeLsum"]),
        "bertscore.precision": float(bertscore_results["precision"]),
        "bertscore.recall": float(bertscore_results["recall"]),
        "bertscore.f1": float(bertscore_results["f1"]),
    }


def dcg_at_k(relevances, k):
    relevances = np.asarray(relevances)[:k]
    positions = np.arange(1, len(relevances) + 1)
    return np.sum(relevances / np.log2(positions + 1))


def dcg_at_k_pytorch(relevances, k, device):
    relevances = relevances[:k]
    positions = torch.arange(1, len(relevances) + 1, dtype=torch.float32).to(device)
    return torch.sum(relevances / torch.log2(positions + 1))


def idcg_at_k(relevances, k):
    sorted_relevances = sorted(relevances, reverse=True)
    return dcg_at_k(sorted_relevances, k)


def ndcg_at_k(relevances, k):
    dcg = dcg_at_k(relevances, k)
    idcg = idcg_at_k(relevances, k)
    return dcg / idcg if idcg > 0 else 0


def ndcg_at_k_pytorch(relevances, k, device):
    dcg = dcg_at_k_pytorch(relevances, k, device)
    ideal_relevances = torch.sort(relevances, descending=True).values
    idcg = dcg_at_k_pytorch(ideal_relevances, k, device)
    return dcg / idcg if idcg > 0 else torch.tensor(0.0).to(device)


def calculate_ndcg(df, aspect, predictions, config):
    k = config.k
    users = df['user_id'].unique()
    
    ndcg_scores = []
    
    for user in users:
        user_df = df[df['user_id'] == user]
        user_predictions = [pred for idx, pred in enumerate(predictions) if df.iloc[idx]['user_id'] == user]
        
        user_df.loc[:, 'predicted'] = user_predictions
        user_df = user_df.sort_values(by='predicted', ascending=False)
        
        relevances = user_df[aspect].values
        
        ndcg = ndcg_at_k(relevances, k)
        ndcg_scores.append(ndcg)
    
    mean_ndcg = np.mean(ndcg_scores)
    
    return mean_ndcg


def precision_at_k(relevances, k):
    relevances = np.asarray(relevances)[:k]
    return np.mean(relevances)


def average_precision(relevances, threshold_rating):
    relevances = np.asarray(relevances)
    precisions = []
    num_relevant = 0
    
    for k in range(1, len(relevances) + 1):
        if relevances[k - 1] >= threshold_rating:
            num_relevant += 1
            precisions.append(num_relevant / k)
    
    if num_relevant == 0:
        return 0
    
    return np.mean(precisions)


def average_precision_pytorch(relevances, threshold_rating, device):
    precisions = []
    num_relevant = 0
    for k in range(1, len(relevances) + 1):
        if relevances[k - 1] >= threshold_rating:
            num_relevant += 1
            precisions.append(num_relevant / k)

    if num_relevant == 0:
        return torch.tensor(0.0).to(device)
    
    return torch.tensor(precisions).mean().to(device)


def calculate_map(df, aspect, predictions, config):
    users = df['user_id'].unique()
    average_precisions = []
    
    for user in users:
        user_df = df[df['user_id'] == user].copy()
        user_predictions = [pred for idx, pred in enumerate(predictions) if df.iloc[idx]['user_id'] == user]
        
        user_df.loc[:, 'predicted'] = user_predictions
        user_df = user_df.sort_values(by='predicted', ascending=False)
        
        relevances = user_df[aspect].values
        
        ap = average_precision(relevances, config.threshold_rating)
        average_precisions.append(ap)
    
    mean_map = np.mean(average_precisions)
    
    return mean_map


def calculate_mrr(df, aspect, predictions, config):
    users = df['user_id'].unique()
    reciprocal_ranks = []
    
    for user in users:
        user_df = df[df['user_id'] == user].copy()
        user_predictions = [pred for idx, pred in enumerate(predictions) if df.iloc[idx]['user_id'] == user]
        
        user_df.loc[:, 'predicted'] = user_predictions
        user_df = user_df.sort_values(by='predicted', ascending=False)
        
        relevances = user_df[aspect].values
        
        for rank, relevance in enumerate(relevances, start=1):
            if relevance >= config.threshold_rating:
                reciprocal_ranks.append(1 / rank)
                break
        else:
            reciprocal_ranks.append(0)
    
    mean_mrr = np.mean(reciprocal_ranks)
    
    return mean_mrr
