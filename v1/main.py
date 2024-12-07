# Ben Kabongo
# November 2024

from data import *
from evaluation import *
from model import *
from utils import *


def train(model: AURA, config, optimizer, dataloader, training_phase=0):
    model.train()

    losses = {"total": 0.0}
    if training_phase == 0:
        losses.update({"rating": 0.0, "overall_rating": 0.0, "aspect_rating": 0.0, "review_rating": 0.0})
    elif training_phase == 1:
        losses.update({"review": 0.0})

    for batch in tqdm(dataloader, "Training {training_phase}", colour="cyan", total=len(dataloader)):
        U_ids = torch.LongTensor(batch["user_id"]).to(config.device) # (batch_size,)
        U_document_tokens = torch.LongTensor(batch["user_document_tokens"]).to(config.device) # (batch_size, n_sentences, sentence_length)
        U_masks = torch.LongTensor(batch["user_document_masks"]).to(config.device) # (batch_size, n_sentences, sentence_length)
        U_ratings = torch.tensor(batch["user_rating"], dtype=torch.float32).to(config.device) # (batch_size,)
        
        I_ids = torch.LongTensor(batch["item_id"]).to(config.device) # (batch_size,)
        I_document_tokens = torch.LongTensor(batch["item_document_tokens"]).to(config.device) # (batch_size, n_sentences, sentence_length)
        I_masks = torch.LongTensor(batch["item_document_masks"]).to(config.device) # (batch_size, n_sentences, sentence_length)
        I_ratings = torch.tensor(batch["item_rating"], dtype=torch.float32).to(config.device) # (batch_size,)

        if training_phase == 0:
            R = torch.tensor(batch["overall_rating"], dtype=torch.float32).to(config.device) # (batch_size,)
            A_ratings = torch.tensor(batch["aspects_ratings"], dtype=torch.float32).to(config.device) # (batch_size, n_aspects)
            UI_reviews_tokens = None
        
        elif training_phase == 1:
            R = None
            A_ratings = None
            UI_reviews_tokens = torch.LongTensor(batch["review_tokens"]).to(config.device) # (batch_size, review_length)

        output = model(
            U_ids, I_ids,
            U_document_tokens, U_masks, U_ratings, 
            I_document_tokens, I_masks, I_ratings,
            R, A_ratings,
            UI_reviews_tokens,
            inference_flag=False
        )

        loss = output["losses"]["total"]
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
    for key in ["overall_rating", "aspect_rating", "user_rating", "item_rating", "review",
               *([f"{aspect}_rating" for aspect in config.aspects])]:
        references[key] = []
        predictions[key] = []

    for batch_idx, batch in tqdm(enumerate(dataloader), "Evaluation", colour="cyan", total=len(dataloader)):
        U_ids = torch.LongTensor(batch["user_id"]).to(config.device) # (batch_size,)
        U_document_tokens = torch.LongTensor(batch["user_document_tokens"]).to(config.device) # (batch_size, n_sentences, sentence_length)
        U_masks = torch.LongTensor(batch["user_document_masks"]).to(config.device) # (batch_size, n_sentences, sentence_length)
        U_ratings = torch.tensor(batch["user_rating"], dtype=torch.float32).to(config.device) # (batch_size,)
        
        I_ids = torch.LongTensor(batch["item_id"]).to(config.device) # (batch_size,)
        I_document_tokens = torch.LongTensor(batch["item_document_tokens"]).to(config.device) # (batch_size, n_sentences, sentence_length)
        I_masks = torch.LongTensor(batch["item_document_masks"]).to(config.device) # (batch_size, n_sentences, sentence_length)
        I_ratings = torch.tensor(batch["item_rating"], dtype=torch.float32).to(config.device) # (batch_size,)

        R = torch.tensor(batch["overall_rating"], dtype=torch.float32).to(config.device) # (batch_size,)
        A_ratings = torch.tensor(batch["aspects_ratings"], dtype=torch.float32).to(config.device) # (batch_size, n_aspects)
        UI_reviews_tokens = torch.LongTensor(batch["review_tokens"]).to(config.device) # (batch_size, sentence_length)
        reviews = (batch["review"]) # (batch_size,)

        output = model(
            U_ids, I_ids,
            U_document_tokens, U_masks, U_ratings, 
            I_document_tokens, I_masks, I_ratings,
            R, A_ratings,
            UI_reviews_tokens,
            inference_flag=True
        )

        R_hat = output["overall_rating"]
        A_ratings_hat = output["aspects_ratings"]
        reviews_hat = output["review"]

        U_ratings_hat = output["user"]["document"]["review"]["rating"]
        I_ratings_hat = output["item"]["document"]["review"]["rating"]

        users.extend(U_ids.cpu().detach().tolist())

        references["overall_rating"].extend(R.cpu().detach().tolist())
        predictions["overall_rating"].extend(R_hat.cpu().detach().tolist())

        references["review"].extend(reviews)
        predictions["review"].extend(reviews_hat)
            
        references["user_rating"].extend(U_ratings.cpu().detach().tolist())
        predictions["user_rating"].extend(U_ratings_hat.cpu().detach().tolist())
        references["item_rating"].extend(I_ratings.cpu().detach().tolist())
        predictions["item_rating"].extend(I_ratings_hat.cpu().detach().tolist())

        for a, aspect in enumerate(config.aspects):
            references[f"{aspect}_rating"].extend(A_ratings[:, a].cpu().detach().tolist())
            predictions[f"{aspect}_rating"].extend(A_ratings_hat[:, a].cpu().detach().tolist())

        if config.verbose and batch_idx == 0:
            for i in range(len(reviews)):
                log = "\n" + "\n".join([
                    f"User ID: {U_ids[i]}",
                    f"Item ID: {I_ids[i]}",
                    f"Overall Rating: Actual={R[i]:.4f} Predicted={R_hat[i]:4f}",
                    *[f"{aspect} Rating: Actual={A_ratings[i][a]:.4f} Predicted={A_ratings_hat[i][a]:.4f}" 
                    for a, aspect in enumerate(config.aspects)],
                    "Review:",
                    f"\tGround Truth: {reviews[i]}",
                    f"\tGenerated: {reviews_hat[i]}",
                    f"-" * 80
                ])
                print(log)
                with open(config.log_file_path, "w", encoding="utf-8") as log_file:
                    log_file.write(log)

    scores = {}
    for metric in references:
        if metric == "review":
            scores[metric] = review_evaluation(config, predictions[metric], references[metric])
        elif metric in ["user_rating", "item_rating"]:
            scores[metric] = rating_evaluation_pytorch(config, predictions[metric], references[metric])
        else:
            scores[metric] = rating_evaluation_pytorch(config, predictions[metric], references[metric], users)
    return scores


def trainer(model: AURA, config, train_dataloader, eval_dataloader):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

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

        for loss in losses.keys():
            if loss not in train_infos.keys():
                train_infos[loss] = []
            train_infos[loss].append(losses[loss])

        train_loss = losses["total"]
        desc = (
            f"[{epoch} / {config.n_epochs}] Loss: {train_loss:.4f} " +
            f"Best: {config.rating_metric}={best_rating:.4f} {config.review_metric}={best_review:.4f}"
        )

        if epoch % config.eval_every == 0:
            with torch.no_grad():
                scores = eval(model, config, dataloader=eval_dataloader)
            
            for metric_set in scores.keys():
                if metric_set not in eval_infos.keys():
                    eval_infos[metric_set] = {}
                for metric in scores[metric_set].keys():
                    if metric not in eval_infos[metric_set].keys():
                        eval_infos[metric_set][metric] = []
                    eval_infos[metric_set][metric].append(scores[metric_set][metric])

            eval_rating = scores["overall_rating"][config.rating_metric]
            eval_review = scores["review"][config.review_metric]

            if training_phase == 0:
                if eval_rating < best_rating:
                    save_model(model, config.save_model_path)
                    best_rating = eval_rating

            elif training_phase == 1:
                if eval_review > best_review:
                    save_model(model, config.save_model_path)
                    best_review = eval_review

            desc = (
                f"[{epoch} / {config.n_epochs}] " +
                f"Loss: train={train_loss:.4f} " +
                f"Test: {config.rating_metric}={eval_rating:.4f} {config.review_metric}={eval_review:.4f} " +
                f"Best: {config.rating_metric}={best_rating:.4f} {config.review_metric}={best_review:.4f}"
            )

        progress_bar.set_description(desc)

        results = {"train": train_infos, "eval": eval_infos}
        with open(config.res_file_path, "w") as res_file:
            json.dump(results, res_file)

    return train_infos, eval_infos


def run(config):
    set_seed(config.seed)

    if config.dataset_dir == "":
        config.dataset_dir = os.path.join(config.base_dir, config.dataset_name)
    if config.dataset_path == "":
        config.dataset_path = os.path.join(config.dataset_dir, "data.csv")

    data_df = pd.read_csv(config.dataset_path)
    (train_df, eval_df, test_df), (users_vocab, items_vocab) = process_data(data_df, config)

    config.n_users = len(users_vocab)
    config.n_items = len(items_vocab)
    config.n_aspects = len(config.aspects)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device = device

    tokenizer = T5Tokenizer.from_pretrained(config.model_name_or_path)
    train_dataset = RatingsReviewDataset(train_df, config, tokenizer)
    eval_dataset = RatingsReviewDataset(eval_df, config, tokenizer)
    test_dataset = RatingsReviewDataset(test_df, config, tokenizer)

    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn
    )    

    t5_model = T5ForConditionalGeneration.from_pretrained(config.model_name_or_path)
    if config.prompt_tuning_flag:
        for param in t5_model.parameters():
            param.requires_grad = False
    text_module = T5TextModule(config, t5_model)

    model = AURA(config, text_module=text_module, tokenizer=tokenizer)
    model.to(config.device)
    if config.save_model_path != "":
        load_model(model, config.save_model_path)

    exps_base_dir = os.path.join(config.dataset_dir, "exps")
    exp_dir = os.path.join(exps_base_dir, config.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    config.exp_dir = exp_dir
    config.log_file_path = os.path.join(exp_dir, "log.txt")
    config.res_file_path = os.path.join(exp_dir, "res.json")

    if config.save_model_path == "":
        config.save_model_path = os.path.join(exp_dir, "model.pth")

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
        print(log)
        with open(config.log_file_path, "w", encoding="utf-8") as log_file:
            log_file.write(log)

    train_infos, eval_infos = trainer(model, config, train_dataloader, eval_dataloader)
    load_model(model, config.save_model_path)
    with torch.no_grad():
        test_infos = eval(model, config, dataloader=test_dataloader)

    results = {"test": test_infos, "train": train_infos, "eval": eval_infos}
    with open(config.res_file_path, "w") as res_file:
        json.dump(results, res_file)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default=os.path.join("datasets", "processed"))
    parser.add_argument("--dataset_dir", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="Hotels")
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--aspects", type=list, default=[])
    parser.add_argument("--aspects_sep", type=str, default=" ")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--exp_name", type=str, default="test")

    parser.add_argument("--review_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(review_flag=True)
    parser.add_argument("--rating_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(rating_flag=False)

    parser.add_argument("--model_name_or_path", type=str, default="t5-small")
    parser.add_argument("--d_words", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_sentences", type=int, default=100)
    parser.add_argument("--sentence_length", type=int, default=64)
    parser.add_argument("--review_length", type=int, default=256)
    parser.add_argument("--n_prompt", type=int, default=32)
    parser.add_argument("--prompt_tuning_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(prompt_tuning_flag=True)
    parser.add_argument("--save_model_path", type=str, default="")

    parser.add_argument("--alpha", type=int, default=1)
    parser.add_argument("--beta", type=int, default=1)
    parser.add_argument("--gamma", type=int, default=1)
    parser.add_argument("--lambda_", type=int, default=1)

    parser.add_argument("--n_ratings_epochs", type=int, default=20)
    parser.add_argument("--n_reviews_epochs", type=int, default=20)
    parser.add_argument("--rating_metric", type=str, default="rmse")
    parser.add_argument("--review_metric", type=str, default="blue")
    parser.add_argument("--lr", type=float, default=1e-3)
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

    config = parser.parse_args()
    run(config)
