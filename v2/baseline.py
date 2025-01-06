# Ben Kabongo
# December 2024

import logging
from torch.utils.tensorboard import SummaryWriter

from architectures import *
from data import *
from evaluation import *
from module import *
from utils import *


class MLPMultiAspects(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Rating
        self.ratings_loss = RatingsLoss(config)

        self.user_embedding = nn.Embedding(config.n_users, config.d_model)
        self.item_embedding = nn.Embedding(config.n_items, config.d_model)

        self.aspects_rating = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * config.d_model, config.d_model),
                nn.ReLU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(config.d_model, 1)
            ) for _ in range(config.n_aspects)
        ])
        self.overall_rating = nn.Sequential(
            nn.Linear(2 * config.d_model, config.d_model),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(config.d_model, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, U_ids: torch.Tensor, I_ids: torch.Tensor,
                R: torch.Tensor=None, A_ratings: torch.Tensor=None) -> Dict[str, torch.Tensor]:
        # U_ids, I_ids: (batch_size,)
        # R: (batch_size,)
        # A_ratings: (batch_size, n_aspects)

        U_embeddings = self.user_embedding(U_ids) # (batch_size, d_model)
        I_embeddings = self.item_embedding(I_ids) # (batch_size, d_model)

        _out = {}
        A_ratings_hat = []
        for i in range(self.config.n_aspects):
            a_rating = self.aspects_rating[i](torch.cat([U_embeddings, I_embeddings], dim=-1)) # (batch_size,)
            #a_rating = torch.clamp(a_rating, min=self.config.min_rating, max=self.config.max_rating)
            A_ratings_hat.append(a_rating)

        A_ratings_hat = torch.stack(A_ratings_hat, dim=1).squeeze(2) # (batch_size, n_aspects)
        R_hat = self.overall_rating(torch.cat([U_embeddings, I_embeddings], dim=-1)).squeeze(1)
        #R_hat = torch.clamp(R_hat, min=self.config.min_rating, max=self.config.max_rating)
        losses = self.ratings_loss(R, R_hat, A_ratings, A_ratings_hat)
        _out.update({
            "overall_rating": R_hat,
            "aspects_ratings": A_ratings_hat,
            "losses": losses
        })

        return _out


def train(model: MLPMultiAspects, config, optimizer, dataloader):
    model.train()

    losses = {"total": 0.0, "overall_rating": 0.0, "aspect_rating": 0.0}
    for batch in tqdm(dataloader, f"Training", colour="cyan", total=len(dataloader)):
        U_ids = torch.LongTensor(batch["user_id"]).to(config.device) # (batch_size,)
        I_ids = torch.LongTensor(batch["item_id"]).to(config.device) # (batch_size,)
        R = torch.tensor(batch["overall_rating"], dtype=torch.float32).to(config.device) # (batch_size,)
        A_ratings = torch.tensor(batch["aspects_ratings"], dtype=torch.float32).to(config.device) # (batch_size, n_aspects)
        output = model(U_ids, I_ids, R, A_ratings)
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
    for key in ["overall_rating", "aspect_rating",
               *([f"{aspect}_rating" for aspect in config.aspects])]:
        references[key] = []
        predictions[key] = []

    for batch_idx, batch in tqdm(enumerate(dataloader), "Evaluation", colour="cyan", total=len(dataloader)):
        U_ids = torch.LongTensor(batch["user_id"]).to(config.device) # (batch_size,)
        I_ids = torch.LongTensor(batch["item_id"]).to(config.device) # (batch_size,)
        R = torch.tensor(batch["overall_rating"], dtype=torch.float32).to(config.device) # (batch_size,)
        A_ratings = torch.tensor(batch["aspects_ratings"], dtype=torch.float32).to(config.device) # (batch_size, n_aspects)
        output = model(U_ids, I_ids, R, A_ratings)
        R_hat = output["overall_rating"]
        A_ratings_hat = output["aspects_ratings"]

        users.extend(U_ids.cpu().detach().tolist())
        references["overall_rating"].extend(R.cpu().detach().tolist())
        predictions["overall_rating"].extend(R_hat.cpu().detach().tolist())
        for a, aspect in enumerate(config.aspects):
            references[f"{aspect}_rating"].extend(A_ratings[:, a].cpu().detach().tolist())
            predictions[f"{aspect}_rating"].extend(A_ratings_hat[:, a].cpu().detach().tolist())

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
                config.logger.info(log)

    scores = {}
    for element in references:
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

    progress_bar = tqdm(range(1, 1 + config.n_epochs), "Training", colour="blue")
    for epoch in progress_bar:
        losses = train(model, config, optimizer=optimizer, dataloader=train_dataloader)
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
            if eval_rating < best_rating:
                save_model(model, config.save_model_path)
                best_rating = eval_rating

            desc = (
                f"[{epoch} / {config.n_epochs}] " +
                f"Loss: train={train_loss:.4f} " +
                f"Rating ({config.rating_metric}): test={eval_rating:.4f} best={best_rating:.4f}"
            )

        config.logger.info(desc)
        progress_bar.set_description(desc)

        results = {"train": train_infos, "eval": eval_infos}
        #config.logger.info(results)
        with open(config.res_file_path, "w") as res_file:
            json.dump(results, res_file)

        #scheduler.step()

    return train_infos, eval_infos


def run(config):
    set_seed(config.seed)

    os.makedirs(config.save_dir, exist_ok=True)
    config.log_file_path = os.path.join(config.save_dir, "log.txt")
    config.res_file_path = os.path.join(config.save_dir, "res.json")
    config.save_model_path = os.path.join(config.save_dir, "model.pth")

    logger = logging.getLogger("Baseline")
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
    train_dataset = RatingsReviewDataset(train_df, config, tokenizer)
    eval_dataset = RatingsReviewDataset(eval_df, config, tokenizer)
    test_dataset = RatingsReviewDataset(test_df, config, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    eval_dataloader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)    

    model = MLPMultiAspects(config)
    model.to(config.device)
    if config.load_model:
        load_model(model, config.save_model_path)

    if config.verbose:
        log = "\n" + (
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

    parser.add_argument("--review_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(review_flag=False)
    parser.add_argument("--rating_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(rating_flag=True)
    parser.add_argument("--min_rating", type=float, default=1.0)
    parser.add_argument("--max_rating", type=float, default=5.0)

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--load_model", action=argparse.BooleanOptionalAction)
    parser.set_defaults(load_model=False)

    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--beta", type=float, default=1)

    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--rating_metric", type=str, default="rmse")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=512)
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

    config = parser.parse_args()
    run(config)
