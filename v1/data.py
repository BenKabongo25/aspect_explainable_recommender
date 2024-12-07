# Ben Kabongo
# November 2024

from utils import *
    

class RatingsReviewDataset(Dataset):
    
        def __init__(self, data_df: pd.DataFrame, config: Any, tokenizer: T5Tokenizer):
            super().__init__()
            self.data_df = data_df
            self.tokenizer = tokenizer
            self.config = config

            self.users = self.data_df["user_id"].unique().tolist()
            self.items = self.data_df["item_id"].unique().tolist()

            self.users_index = {user_id: [] for user_id in self.users}
            self.items_index = {item_id: [] for item_id in self.items}

            self.ratings = self.data_df["rating"].tolist()
            self.reviews = self.data_df["review"].tolist()

            self.reviews_tokens = []
            self.reviews_masks = []

            self.sentences = []
            self.sentences_tokens = []
            self.sentences_masks = []
            self.sentence_tokens_pad = torch.full((self.config.sentence_length,), self.tokenizer.pad_token_id, dtype=torch.long)
            self.sentence_masks_pad = torch.zeros((self.config.sentence_length,), dtype=torch.long)

            self._process()
    
        def __len__(self) -> int:
            return len(self.data_df)
        
        def _process(self):
            for index in tqdm(range(len(self)), desc="Processing data", colour="green"):
                row = self.data_df.iloc[index]

                user_id = row["user_id"]
                item_id = row["item_id"]

                self.users_index[user_id].append(index)
                self.items_index[item_id].append(index)

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
                masks = inputs["attention_mask"].squeeze(0)
                self.reviews_tokens.append(tokens)
                self.reviews_masks.append(masks)

                sentences = sent_tokenize(review)
                sentences = sentences[:min(len(sentences), self.config.n_sentences)]
                self.sentences.append(sentences)

                sentences_tokens = []
                sentences_masks = []
                for sentence in sentences:
                    inputs = self.tokenizer(
                        sentence, 
                        max_length=self.config.sentence_length,
                        truncation=True, 
                        padding="max_length",
                        return_tensors="pt"
                    )
                    tokens = inputs["input_ids"].squeeze(0)
                    masks = inputs["attention_mask"].squeeze(0)
                    sentences_tokens.append(tokens)
                    sentences_masks.append(masks)
                self.sentences_tokens.append(sentences_tokens)
                self.sentences_masks.append(sentences_masks)
                
        def __getitem__(self, index) -> Any:
            random.seed(index)
            row = self.data_df.iloc[index]

            user_id = row["user_id"]
            item_id = row["item_id"]

            overall_rating = row["rating"]
            aspects_ratings = [row[aspect] for aspect in self.config.aspects]

            review = self.reviews[index]
            review_tokens = self.reviews_tokens[index].clone()
            review_tokens[review_tokens == self.tokenizer.pad_token_id] = -100

            user_reviews_index = list(self.users_index[user_id])
            user_reviews_index.remove(index)
            user_review_index = random.choice(user_reviews_index)
            user_document = self.sentences[user_review_index]
            user_document_tokens = self.sentences_tokens[user_review_index]
            user_document_masks = self.sentences_masks[user_review_index]
            if len(user_document) < self.config.n_sentences:
                pad_length = self.config.n_sentences - len(user_document)
                user_document_tokens.extend([self.sentence_tokens_pad.clone() for _ in range(pad_length)])
                user_document_masks.extend([self.sentence_masks_pad.clone() for _ in range(pad_length)])
            user_rating = self.ratings[user_review_index]

            item_reviews_index = list(self.items_index[item_id])
            item_reviews_index.remove(index)
            item_review_index = random.choice(item_reviews_index)
            item_document = self.sentences[item_review_index]
            item_document_tokens = self.sentences_tokens[item_review_index]
            item_document_masks = self.sentences_masks[item_review_index]
            if len(item_document) < self.config.n_sentences:
                pad_length = self.config.n_sentences - len(item_document)
                item_document_tokens.extend([self.sentence_tokens_pad.clone() for _ in range(pad_length)])
                item_document_masks.extend([self.sentence_masks_pad.clone() for _ in range(pad_length)])
            item_rating = self.ratings[item_review_index]

            random.seed(self.config.seed)
                        
            return {
                "user_id": user_id,
                "item_id": item_id,
                "user_document_tokens": user_document_tokens,
                "item_document_tokens": item_document_tokens,
                "user_document_masks": user_document_masks,
                "item_document_masks": item_document_masks,
                "user_rating": user_rating,
                "item_rating": item_rating,
                "overall_rating": overall_rating,
                "aspects_ratings": aspects_ratings,
                "review": review,
                "review_tokens": review_tokens
            }
        
        
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
