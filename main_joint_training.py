# Ben Kabongo
# January 2025

# AURA
# Joint training

import argparse
import json
import logging
import pandas as pd
import os

from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
from typing import Union


from data import *
from evaluation import *
from modules import *
from utils import *


def write_loss(writer, losses, phase, epoch):
    for loss in losses:
        writer.add_scalar(f"{loss}/{phase}", losses[loss], epoch)


def write_eval(writer, scores, phase, epoch):
    for element in scores:
        for metric in scores[element]:
            writer.add_scalar(f"{element}/{metric}/{phase}", scores[element][metric], epoch)


def train(
        config,
        model: Union[AURA, AURA_A],
        rating_loss_fn: Union[nn.MSELoss, AspectsRatingLoss],
        optimizer: Optimizer, 
        dataloader: DataLoader
    ):
    model.train()
    runing_losses = {"total": 0.0, "overall_rating": 0.0, "aspects_ratings": 0.0, "review": 0.0}

    for batch in tqdm(dataloader, f"Train", colour="cyan", total=len(dataloader)):
        U_ids = torch.LongTensor(batch["user_id"]).to(config.device) # (batch_size,)
        I_ids = torch.LongTensor(batch["item_id"]).to(config.device) # (batch_size,)
        labels = torch.LongTensor(batch["review_labels"]).to(config.device) # (batch_size, review_length)
        output = model(U_ids, I_ids, labels)

        R = torch.tensor(batch["overall_rating"], dtype=torch.float32).to(config.device) # (batch_size,)
        R_hat = output.rating_module.overall_rating # (batch_size,)
        
        if config.aspects_flag:
            A_ratings = torch.tensor(batch["aspects_ratings"], dtype=torch.float32).to(config.device) # (batch_size, n_aspects)
            A_ratings_hat = output.rating_module.aspects_ratings # (batch_size, n_aspects)
            rating_losses = rating_loss_fn(R, R_hat, A_ratings, A_ratings_hat)
            rating_loss = rating_losses.total

            runing_losses["overall_rating"] += rating_losses.overall_rating.item()
            runing_losses["aspects_ratings"] += rating_losses.aspects_ratings.item()

        else:
            rating_loss = rating_loss_fn(R, R_hat)
            runing_losses["overall_rating"] += rating_loss.item()

        review_loss = output.review_module.loss
        runing_losses["review"] += review_loss.item()

        loss = config.lambda_rating * rating_loss + config.lambda_review * review_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for loss in runing_losses.keys():
        runing_losses[loss] /= len(dataloader)
    return runing_losses


def eval(
        config,
        model: Union[AURA, AURA_A],
        dataloader: DataLoader,
        rating_metrics: List[str]=[RMSE],
        review_metrics: List[str]=[BLEU, METEOR]
    ):
    model.eval()

    users = []
    references = {"overall_rating": [], "review": []}
    predictions = {"overall_rating": [], "review": []}

    if config.aspects_flag:
        for aspect in config.aspects:
            references[f"{aspect}_rating"] = []
            predictions[f"{aspect}_rating"] = []
           
    for batch_idx, batch in tqdm(enumerate(dataloader), "Eval", colour="cyan", total=len(dataloader)):
        U_ids = torch.LongTensor(batch["user_id"]).to(config.device) # (batch_size,)
        I_ids = torch.LongTensor(batch["item_id"]).to(config.device) # (batch_size,)
        reviews = (batch["review"]) # (batch_size,)
        
        output = model(U_ids, I_ids)

        R = torch.tensor(batch["overall_rating"], dtype=torch.float32).to(config.device) # (batch_size,)
        R_hat = output.rating_module.overall_rating # (batch_size,)

        users.extend(U_ids.cpu().detach().tolist())
        references["overall_rating"].extend(R.cpu().detach().tolist())
        predictions["overall_rating"].extend(R_hat.cpu().detach().tolist())
        
        if config.aspects_flag:
            A_ratings = torch.tensor(batch["aspects_ratings"], dtype=torch.float32).to(config.device) # (batch_size, n_aspects)
            A_ratings_hat = output.rating_module.aspects_ratings # (batch_size, n_aspects)

            for a, aspect in enumerate(config.aspects):
                references[f"{aspect}_rating"].extend(A_ratings[:, a].cpu().detach().tolist())
                predictions[f"{aspect}_rating"].extend(A_ratings_hat[:, a].cpu().detach().tolist())

        reviews_hat = model.generate(U_ids, I_ids)
        references["review"].extend(reviews)
        predictions["review"].extend(reviews_hat)

        if config.verbose and batch_idx == 0:
            n_samples = min(5, len(U_ids))
            for i in range(n_samples):
                log = "\n" + "\n".join([
                    f"User ID: {U_ids[i]} ",
                    f"Item ID: {I_ids[i]} ",
                    f"Overall Rating: Actual={R[i]:.4f} Predicted={R_hat[i]:4f}\n",
                ])
                if config.aspects_flag:
                    log += "\n".join([
                        *[f"{aspect} Rating: Actual={A_ratings[i][a]:.4f} Predicted={A_ratings_hat[i][a]:.4f}" 
                        for a, aspect in enumerate(config.aspects)]
                    ])
                log += "\n" + "\n".join([
                    f"Review: {reviews[i]}",
                    f"Generated: {reviews_hat[i]}"
                ])
                config.logger.info(log)

    scores = {}
    for element in references:
        if element == "review":
            scores[element] = review_evaluation(config, predictions[element], references[element], review_metrics)
        else:
            scores[element] = rating_evaluation(config, predictions[element], references[element], users, rating_metrics)
    return scores


def trainer(
        config,
        model: Union[AURA, AURA_A],
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader
    ):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    if config.aspects_flag:
        rating_loss_fn = AspectsRatingLoss(config)
    else:
        rating_loss_fn = nn.MSELoss()

    train_infos = {}
    eval_infos = {}

    best_rating = float("inf")
    rating_metrics = [config.rating_metric]
    best_review = -float("inf")
    review_metrics = [config.review_metric]

    progress_bar = tqdm(range(1, 1 + config.n_epochs), "Training", colour="blue")
    for epoch in progress_bar:
        losses = train(config, model, rating_loss_fn, optimizer, train_dataloader)
        write_loss(config.writer, losses, "train", epoch)

        for loss in losses.keys():
            if loss not in train_infos.keys():
                train_infos[loss] = []
            train_infos[loss].append(losses[loss])

        train_loss = losses["total"]
        desc = (
            f"[{epoch} / {config.n_epochs}] Loss: {train_loss:.4f} " +
            f"Rating ({config.rating_metric}): best={best_rating:.4f} " +
            f"Review ({config.review_metric}): best={best_review:.4f}"
        )

        if epoch % config.eval_every == 0:
            with torch.no_grad():
                scores = eval(config, model, eval_dataloader, rating_metrics, review_metrics)
            write_eval(config.writer, scores, "eval", epoch)
            
            for metric_set in scores.keys():
                if metric_set not in eval_infos.keys():
                    eval_infos[metric_set] = {}
                for metric in scores[metric_set].keys():
                    if metric not in eval_infos[metric_set].keys():
                        eval_infos[metric_set][metric] = []
                    eval_infos[metric_set][metric].append(scores[metric_set][metric])

            eval_rating = scores["overall_rating"][config.rating_metric]
            eval_review = scores["review"][config.review_metric]
            if eval_rating < best_rating and eval_review > best_review:
                model.save(config.save_model_path)
                best_rating = eval_rating
                best_review = eval_review

            desc = (
                f"[{epoch} / {config.n_epochs}] " +
                f"Loss: train={train_loss:.4f} " +
                f"Rating ({config.rating_metric}): test={eval_rating:.4f} best={best_rating:.4f} " +
                f"Review ({config.review_metric}): test={eval_review:.4f} best={best_review:.4f}"
            )

        config.logger.info(desc)
        progress_bar.set_description(desc)

        results = {"train": train_infos, "eval": eval_infos}
        with open(config.res_file_path, "w") as res_file:
            json.dump(results, res_file)

    return train_infos, eval_infos
    

def run(config):
    set_seed(config.seed)

    os.makedirs(config.save_dir, exist_ok=True)
    config.log_file_path = os.path.join(config.save_dir, "log.txt")
    config.res_file_path = os.path.join(config.save_dir, "res.json")
    config.save_model_path = os.path.join(config.save_dir, "model.pth")

    logger = logging.getLogger("AURA")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(config.log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    config.logger = logger

    writer = SummaryWriter(config.save_dir)
    config.writer = writer

    data_df = pd.read_csv(config.dataset_path).drop_duplicates()
    (train_df, eval_df, test_df), (users_vocab, items_vocab) = process_data(config, data_df)

    config.n_users = len(users_vocab)
    config.n_items = len(items_vocab)
    config.n_aspects = len(config.aspects)
    config.n_prompt_elements = 2
    if config.aspects_flag:
        config.n_prompt_elements += 2 * config.n_aspects

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device = device

    tokenizer = None
    text_module = None
    if config.review_flag:
        tokenizer = T5Tokenizer.from_pretrained(config.model_name_or_path)
        t5_model = T5ForConditionalGeneration.from_pretrained(config.model_name_or_path)
        if config.prompt_tuning_flag:
            for param in t5_model.parameters():
                param.requires_grad = False
        text_module = T5TextModule(config, t5_model, tokenizer)
        config.n_reviews_epochs = 0

    train_dataset = RatingsReviewDataset(config, train_df, tokenizer)
    eval_dataset = RatingsReviewDataset(config, eval_df, tokenizer)
    test_dataset = RatingsReviewDataset(config, test_df, tokenizer)

    log = "\n" + (
        f"Dataset: {config.dataset_name}\n" +
        f"Aspects: {config.aspects}\n" +
        f"#Users: {config.n_users}\n" +
        f"#Items: {config.n_items}\n" +
        f"Device: {device}\n\n" +
        f"Args:\n{config}\n\n" +
        f"Data:\n{train_df.head(5)}\n\n"
    )
    config.logger.info(log)

    train_dataloader = DataLoader(train_dataset, batch_size=config.rating_batch_size, shuffle=True, collate_fn=collate_fn)
    eval_dataloader = DataLoader(eval_dataset, batch_size=config.rating_batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=config.rating_batch_size, shuffle=False, collate_fn=collate_fn)    
    
    review_model = ReviewGenerationModule(config, text_module)
    if config.aspects_flag:
        rating_model = AspectsRatingPredictionModule(config)
        model = AURA(config, rating_model, review_model)
    else:
        rating_model = RatingPredictionModule(config)
        model = AURA_A(config, rating_model, review_model)
    model.to(config.device)

    if config.load_model:
        model.load(config.save_model_path)

    config.logger.info("Training...")
    train_infos, eval_infos = trainer(config, model, rating_model, train_dataloader, eval_dataloader)

    config.logger.info("Testing...")
    review_model.load(config.save_review_model_path)
    with torch.no_grad():
        test_infos = eval(config, model, test_dataloader, RATING_METRICS, REVIEW_METRICS)
    
    results = {"test": test_infos, "train": train_infos, "eval": eval_infos}
    config.logger.info(results)
    with open(config.res_file_path, "w") as res_file:
        json.dump(results, res_file)

    config.logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--aspects_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(aspects_flag=True)
    parser.add_argument("--aspects", type=str, nargs="+")
    parser.add_argument("--lang", type=str, default="en")

    parser.add_argument("--review_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(review_flag=True)
    parser.add_argument("--prompt_tuning_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(prompt_tuning_flag=True)
    parser.add_argument("--n_prompt_tokens", type=int, default=50)
    parser.add_argument("--min_rating", type=float, default=1.0)
    parser.add_argument("--max_rating", type=float, default=5.0)

    parser.add_argument("--model_name_or_path", type=str, default="t5-small")
    parser.add_argument("--d_words", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--review_length", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=0) 
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--load_model", action=argparse.BooleanOptionalAction)
    parser.set_defaults(load_model=False)

    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--beta", type=float, default=1)
    parser.add_argument("--lambda_rating", type=float, default=1)
    parser.add_argument("--lambda_review", type=float, default=1)

    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--rating_metric", type=str, default="rmse")
    parser.add_argument("--review_metric", type=str, default="meteor")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_size", type=float, default=0.8)
    parser.add_argument("--eval_size", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold_rating", type=float, default=4.0)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction)
    parser.set_defaults(verbose=True)
    parser.add_argument("--verbose_every", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=1)

    parser.add_argument("--truncate_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(truncate_flag=True)
    parser.add_argument("--lower_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(lower_flag=True)
    parser.add_argument("--delete_balise_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(delete_balise_flag=False)
    parser.add_argument("--delete_non_ascii_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(delete_non_ascii_flag=True)
    parser.add_argument("--delete_digit_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(delete_digit_flag=False)

    config = parser.parse_args()
    run(config)
