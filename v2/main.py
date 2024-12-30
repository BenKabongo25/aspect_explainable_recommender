# Ben Kabongo
# November 2024


import logging
from torch.utils.tensorboard import SummaryWriter

from architectures import *
from data import *
from evaluation import *
from module import *
from utils import *


def train(model: AURA, config, optimizer, dataloader, training_phase=0):
    model.train()

    losses = {"total": 0.0}
    if training_phase == 0:
        losses.update({"overall_rating": 0.0, "aspect_rating": 0.0})
    elif training_phase == 1:
        losses.update({"review": 0.0})

    for batch in tqdm(dataloader, f"Training {training_phase}", colour="cyan", total=len(dataloader)):
        U_ids = torch.LongTensor(batch["user_id"]).to(config.device) # (batch_size,)
        I_ids = torch.LongTensor(batch["item_id"]).to(config.device) # (batch_size,)

        if not config.review_flag or training_phase == 0:
            R = torch.tensor(batch["overall_rating"], dtype=torch.float32).to(config.device) # (batch_size,)
            A_ratings = torch.tensor(batch["aspects_ratings"], dtype=torch.float32).to(config.device) # (batch_size, n_aspects)
            reviews_tokens = None
        
        else:
            R = None
            A_ratings = None
            reviews_tokens = torch.LongTensor(batch["review_tokens"]).to(config.device) # (batch_size, review_length)

        output = model(U_ids, I_ids, R, A_ratings, reviews_tokens, inference_flag=False)
        loss = output["losses"]["total"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #print("Losses", output["losses"])
        for loss in output["losses"]:
            losses[loss] += output["losses"][loss].item()

    for loss in losses:
        losses[loss] /= len(dataloader)
    return losses
    

def eval(model: AURA, config, dataloader):
    model.eval()

    users = []
    references = {}
    predictions = {}
    for key in ["overall_rating", "aspect_rating", "review",
               *([f"{aspect}_rating" for aspect in config.aspects])]:
        references[key] = []
        predictions[key] = []

    for batch_idx, batch in tqdm(enumerate(dataloader), "Evaluation", colour="cyan", total=len(dataloader)):
        U_ids = torch.LongTensor(batch["user_id"]).to(config.device) # (batch_size,)
        I_ids = torch.LongTensor(batch["item_id"]).to(config.device) # (batch_size,)

        R = torch.tensor(batch["overall_rating"], dtype=torch.float32).to(config.device) # (batch_size,)
        A_ratings = torch.tensor(batch["aspects_ratings"], dtype=torch.float32).to(config.device) # (batch_size, n_aspects)

        reviews_tokens = None
        reviews = None
        if config.review_flag:
            reviews_tokens = torch.LongTensor(batch["review_tokens"]).to(config.device) # (batch_size, sentence_length)
            reviews = (batch["review"]) # (batch_size,)

        output = model(U_ids, I_ids, R, A_ratings, reviews_tokens, inference_flag=True)

        R_hat = output["overall_rating"]
        A_ratings_hat = output["aspects_ratings"]
        reviews_hat = None
        if config.review_flag:
            reviews_hat = output["review"]

        users.extend(U_ids.cpu().detach().tolist())
        references["overall_rating"].extend(R.cpu().detach().tolist())
        predictions["overall_rating"].extend(R_hat.cpu().detach().tolist())
        for a, aspect in enumerate(config.aspects):
            references[f"{aspect}_rating"].extend(A_ratings[:, a].cpu().detach().tolist())
            predictions[f"{aspect}_rating"].extend(A_ratings_hat[:, a].cpu().detach().tolist())
        if config.review_flag:
            references["review"].extend(reviews)
            predictions["review"].extend(reviews_hat)

        if config.verbose and batch_idx == 0:
            n_samples = min(5, len(U_ids))
            for i in range(n_samples):
                log = "\n" + "\n".join([
                    f"User ID: {U_ids[i]}",
                    f"Item ID: {I_ids[i]}",
                    f"Overall Rating: Actual={R[i]:.4f} Predicted={R_hat[i]:4f}",
                    *[f"{aspect} Rating: Actual={A_ratings[i][a]:.4f} Predicted={A_ratings_hat[i][a]:.4f}" 
                    for a, aspect in enumerate(config.aspects)]
                ])
                if config.review_flag:
                    log += "\n" + "\n".join([
                        f"Review: {reviews[i]}",
                        f"Generated: {reviews_hat[i]}"
                    ])
                config.logger.info(log)

    scores = {}
    for element in references:
        if element == "review" and config.review_flag:
            scores[element] = review_evaluation(config, predictions[element], references[element])
        else:
            scores[element] = rating_evaluation_pytorch(config, predictions[element], references[element], users)
    return scores


def write_loss(writer, losses, phase, epoch):
    for loss in losses:
        writer.add_scalar(f"{loss}/{phase}", losses[loss], epoch)


def write_eval(writer, scores, phase, epoch):
    for element in scores:
        for metric in scores[element]:
            writer.add_scalar(f"{element}/{metric}/{phase}", scores[element][metric], epoch)


def trainer(model: AURA, config, train_dataloader, eval_dataloader):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    train_infos = {}
    eval_infos = {}

    best_rating = float("inf")
    best_review = -float("inf")

    config.n_epochs = config.n_ratings_epochs + config.n_reviews_epochs
    training_phase = 0
    model.set_training_phase(training_phase)

    progress_bar = tqdm(range(1, 1 + config.n_epochs), "Training", colour="blue")
    for epoch in progress_bar:

        if epoch > config.n_ratings_epochs:
            training_phase = 1
            model.set_training_phase(training_phase)

        losses = train(model, config, optimizer=optimizer, dataloader=train_dataloader, training_phase=training_phase)
        write_loss(config.writer, losses, "train", epoch)

        for loss in losses.keys():
            if loss not in train_infos.keys():
                train_infos[loss] = []
            train_infos[loss].append(losses[loss])

        train_loss = losses["total"]
        desc = (
            f"[{epoch} / {config.n_epochs}] Loss: {train_loss:.4f} " +
            f"Best: {config.rating_metric}={best_rating:.4f}"
        )
        if config.review_flag:
            desc += f" {config.review_metric}={best_review:.4f}"

        if epoch % config.eval_every == 0:
            with torch.no_grad():
                scores = eval(model, config, dataloader=eval_dataloader)
            write_eval(config.writer, scores, "eval", epoch)
            
            for metric_set in scores.keys():
                if metric_set not in eval_infos.keys():
                    eval_infos[metric_set] = {}
                for metric in scores[metric_set].keys():
                    if metric not in eval_infos[metric_set].keys():
                        eval_infos[metric_set][metric] = []
                    eval_infos[metric_set][metric].append(scores[metric_set][metric])

            eval_rating = scores["overall_rating"][config.rating_metric]
            if config.review_flag:
                eval_review = scores["review"][config.review_metric]

            if training_phase == 0:
                if eval_rating < best_rating:
                    save_model(model, config.save_model_path)
                    best_rating = eval_rating

            elif config.review_flag and training_phase != 0:
                if eval_review > best_review:
                    save_model(model, config.save_model_path)
                    best_review = eval_review

            desc = (
                f"[{epoch} / {config.n_epochs}] " +
                f"Loss: train={train_loss:.4f} " +
                f"Rating ({config.rating_metric}): test={eval_rating:.4f} best={best_rating:.4f}"
            )
            if config.review_flag:
                desc += f" Review ({config.review_metric}): test={eval_review:.4f} best={best_review:.4f}"

        config.logger.info(desc)
        progress_bar.set_description(desc)

        results = {"train": train_infos, "eval": eval_infos}
        #config.logger.info(results)
        with open(config.res_file_path, "w") as res_file:
            json.dump(results, res_file)

        #scheduler.step()

    return train_infos, eval_infos


def get_model(config):
    if config.architecture == "shared_self_attention": # w/ shared self attention
        model = AURASharedSelfAttention
    elif config.architecture == "notshared_self_attention": # w/o shared self attention
        model = AURANotSharedSelfAttention
    elif config.architecture == "shared_cross_attention": # w/ shared cross attention
        model = AURASharedCrossAttention
    elif config.architecture == "notshared_cross_attention": # w/o shared cross attention
        model = AURANotSharedCrossAttention
    elif config.architecture == "skip_connection": # w/ skip connection
        model = AURASkipConnection
    elif config.architecture == "pooling": # w/ pooling
        model = AURAPooling
    elif config.architecture == "shared_aspects_embeddings": # w/ shared aspects embeddings
        model = AURASharedAspectsEmbedding
    elif config.architecture == "shared_aspects_ratings": # w/ shared aspects ratings
        model = AURASharedAspectsRating
    else:
        model = AURA
    return model


def run(config):
    set_seed(config.seed)

    os.makedirs(config.save_dir, exist_ok=True)
    config.log_file_path = os.path.join(config.save_dir, "log.txt")
    config.res_file_path = os.path.join(config.save_dir, "res.json")
    config.save_model_path = os.path.join(config.save_dir, "model.pth")

    logger = logging.getLogger("AURA" + config.architecture)
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

    data_df = pd.read_csv(config.dataset_path).dropna()
    (train_df, eval_df, test_df), (users_vocab, items_vocab) = process_data(data_df, config)

    config.n_users = len(users_vocab)
    config.n_items = len(items_vocab)
    config.n_aspects = len(config.aspects)

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
        text_module = T5TextModule(config, t5_model)
        config.n_reviews_epochs = 0

    train_dataset = RatingsReviewDataset(train_df, config, tokenizer)
    eval_dataset = RatingsReviewDataset(eval_df, config, tokenizer)
    test_dataset = RatingsReviewDataset(test_df, config, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    eval_dataloader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)    

    model = get_model(config)(config, text_module=text_module, tokenizer=tokenizer)
    model.to(config.device)
    if config.load_model:
        load_model(model, config.save_model_path)

    if config.verbose:
        log = "\n" + (
            f"Model name: {config.model_name_or_path}\n" +
            f"Dataset: {config.dataset_name}\n" +
            f"Aspects: {config.aspects}\n" +
            f"#Users: {config.n_users}\n" +
            f"#Items: {config.n_items}\n" +
            f"Device: {device}\n\n" +
            f"Args:\n{config}\n\n" +
            f"Model: {model}\n\n" +
            f"Data:\n{train_df.head(5)}\n\n"
        )
        config.logger.info(log)

    config.logger.info("Training...")
    train_infos, eval_infos = trainer(model, config, train_dataloader, eval_dataloader)

    config.logger.info("Testing...")
    load_model(model, config.save_model_path)
    with torch.no_grad():
        test_infos = eval(model, config, dataloader=test_dataloader)

    results = {"test": test_infos, "train": train_infos, "eval": eval_infos}
    config.logger.info(results)
    with open(config.res_file_path, "w") as res_file:
        json.dump(results, res_file)

    config.logger.info("Done!")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--aspects", type=str, nargs="+")
    parser.add_argument("--lang", type=str, default="en")

    parser.add_argument("--review_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(review_flag=True)
    parser.add_argument("--rating_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(rating_flag=False)
    parser.add_argument("--min_rating", type=float, default=1.0)
    parser.add_argument("--max_rating", type=float, default=5.0)

    parser.add_argument("--model_name_or_path", type=str, default="t5-small")
    parser.add_argument("--d_words", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--review_length", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--load_model", action=argparse.BooleanOptionalAction)
    parser.set_defaults(load_model=False)

    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--beta", type=float, default=1)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--lambda_", type=float, default=1)

    parser.add_argument("--review_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(review_flag=True)
    parser.add_argument("--n_ratings_epochs", type=int, default=20)
    parser.add_argument("--n_reviews_epochs", type=int, default=20)
    parser.add_argument("--rating_metric", type=str, default="rmse")
    parser.add_argument("--review_metric", type=str, default="meteor")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_size", type=float, default=0.8)
    parser.add_argument("--eval_size", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold_rating", type=float, default=4.0)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ranking_metrics_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(ranking_metrics_flag=False)
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--verbose_every", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=1)

    parser.add_argument("--truncate_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(truncate_flag=True)
    parser.add_argument("--lower_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(lower_flag=True)
    parser.add_argument("--delete_balise_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(delete_balise_flag=True)
    parser.add_argument("--delete_stopwords_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(delete_stopwords_flag=False)
    parser.add_argument("--delete_punctuation_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(delete_punctuation_flag=False)
    parser.add_argument("--delete_non_ascii_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(delete_non_ascii_flag=True)
    parser.add_argument("--delete_digit_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(delete_digit_flag=False)
    parser.add_argument("--replace_maj_word_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(replace_maj_word_flag=False)
    parser.add_argument("--first_line_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(first_line_flag=False)
    parser.add_argument("--last_line_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(last_line_flag=False)
    parser.add_argument("--stem_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(stem_flag=False)
    parser.add_argument("--lemmatize_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(lemmatize_flag=False)

    # abllations
    parser.add_argument("--architecture", type=str, default="")
    config = parser.parse_args()
    run(config)
