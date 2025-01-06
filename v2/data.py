# Ben Kabongo
# November 2024

from utils import *
    

class RatingsReviewDataset(Dataset):
    
    def __init__(self, data_df: pd.DataFrame, config: Any, tokenizer: T5Tokenizer=None):
        super().__init__()
        self.data_df = data_df
        self.tokenizer = tokenizer
        self.config = config

        if self.config.review_flag:
            self.reviews = self.data_df["review"].tolist()
            self.reviews_tokens = []
            self._process()
    
    def __len__(self) -> int:
        return len(self.data_df)
        
    def _process(self):
        for index in tqdm(range(len(self)), desc="Processing data", colour="green"):
            review = self.reviews[index]
            review = preprocess_text(review, self.config, self.config.review_length)
            self.reviews[index] = review

            inputs = self.tokenizer(
                review, 
                max_length=self.config.review_length,
                truncation=True, 
                padding="max_length",
                return_tensors="pt"
            )
            tokens = inputs["input_ids"].squeeze(0)
            tokens[tokens == self.tokenizer.pad_token_id] = -100
            self.reviews_tokens.append(tokens)
                
    def __getitem__(self, index) -> Any:
        random.seed(index)
        row = self.data_df.iloc[index]

        user_id = row["user_id"]
        item_id = row["item_id"]

        overall_rating = row["rating"]
        aspects_ratings = [row[aspect] for aspect in self.config.aspects]

        _out = {
            "user_id": user_id,
            "item_id": item_id,
            "overall_rating": overall_rating,
            "aspects_ratings": aspects_ratings
        }

        if not self.config.review_flag:
            return _out

        review = self.reviews[index]
        review_tokens = self.reviews_tokens[index]
        _out.update({"review": review, "review_tokens": review_tokens})
        return _out
        
        
def collate_fn(batch):
    collated_batch = {}
    for key in batch[0]:
        values = [d[key] for d in batch]
        if isinstance(values[0], list) and isinstance(values[0][0], torch.Tensor):
            collated_batch[key] = torch.stack([torch.stack(tensor_list, dim=0) for tensor_list in values], dim=0)
        elif isinstance(values[0], torch.Tensor):
            collated_batch[key] = torch.stack(values, dim=0)
        else:
            collated_batch[key] = values
    return collated_batch
    

def process_data(data_df, config: Any) -> Tuple[pd.DataFrame]:
    data_df['user_id'] = data_df['user_id'].apply(str)
    data_df['item_id'] = data_df['item_id'].apply(str)

    users_vocab = create_vocab_from_df(data_df, 'user_id')
    items_vocab = create_vocab_from_df(data_df, 'item_id')

    data_df['user_id'] = data_df['user_id'].apply(lambda u: to_vocab_id(u, users_vocab))
    data_df['item_id'] = data_df['item_id'].apply(lambda i: to_vocab_id(i, items_vocab))

    train_df = data_df.sample(frac=config.train_size, random_state=config.seed)
    test_eval_df = data_df.drop(train_df.index)
    eval_size = config.eval_size / (config.eval_size + config.test_size)
    eval_df = test_eval_df.sample(frac=eval_size, random_state=config.seed)
    test_df = test_eval_df.drop(eval_df.index)

    return (train_df, eval_df, test_df), (users_vocab, items_vocab)
