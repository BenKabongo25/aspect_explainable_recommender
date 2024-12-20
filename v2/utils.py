# Ben Kabongo
# November 2024

import argparse
import evaluate
import json
import math
import nltk
import numpy as np
import pandas as pd
import os
import random
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import EnglishStemmer
from nltk.tokenize import sent_tokenize
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score, f1_score
from sklearn.preprocessing import Binarizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import Any, Dict, List, Tuple, Union


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def empty_cache():
    with torch.no_grad(): 
        torch.cuda.empty_cache()


class Vocabulary:

    def __init__(self):
        self._elements2ids = {}
        self._ids2elements = {}
        self.n_elements = 0
        self.default_add = True

    def add_element(self, element: Union[int, float, str]):
        if element not in self._elements2ids:
            self._elements2ids[element] = self.n_elements
            self._ids2elements[self.n_elements] = element
            self.n_elements += 1

    def add_elements(self, elements: List[Union[int, float, str]]):
        for element in tqdm(elements, "Vocabulary creation", colour="green"):
            self.add_element(element)

    def __len__(self):
        return self.n_elements
    
    def id2element(self, id: int) -> Union[int, float, str]:
        return self._ids2elements[id]
    
    def element2id(self, element: Union[int, float, str]) -> int:
        if element not in self._elements2ids:
            if self.default_add:
                self.add_element(element)
            else:
                return None
        return self._elements2ids[element]
    
    def ids2elements(self, ids: List[int]) -> List[Union[int, float, str]]:
        return [self._ids2elements[id] for id in ids]
    
    def elements2ids(self, elements: List[Union[int, float, str]]) -> List[int]:
        return [self.element2id(element) for element in elements]

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"elements2ids": self._elements2ids, "ids2elements": self._ids2elements, "n_elements": self.n_elements}, f)

    def load(self, path: str):
        with open(path, "r") as f:
            data = json.load(f)
            self._elements2ids = data["elements2ids"]
            self._ids2elements = data["ids2elements"]
            self.n_elements = data["n_elements"]


def create_vocab_from_df(metadata_df: pd.DataFrame, element_column: str) -> Vocabulary:
    elements = metadata_df[element_column].unique()
    vocab = Vocabulary()
    vocab.add_elements(elements)
    return vocab


def to_vocab_id(element, vocabulary: Vocabulary) -> int:
    return vocabulary.element2id(element)


def to_class_id(rating: float, n_classes: int, args: Any) -> int:
    scale = (args.max_rating - args.min_rating) / n_classes
    c = math.ceil((rating - args.min_rating) / scale) - 1
    if c < 0: c = 0
    if c >= n_classes: c = n_classes - 1
    return c
    

def save_model(model, save_model_path: str):
    torch.save(model.state_dict(), save_model_path)


def load_model(model, save_model_path: str):
    model.load_state_dict(torch.load(save_model_path))


def attention_function(queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    return torch.matmul(F.softmax(torch.matmul(queries, keys.transpose(-2, -1)), dim=-1), values)


def delete_punctuation(text: str) -> str:
    punctuation = r"[\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\\\]\^\_\`\{\|\}\~\n\t]"
    text = re.sub(punctuation, " ", text)
    text = re.sub('( )+', ' ', text)
    return text


def delete_stopwords(text: str) -> str:
    stop_words = set(nltk.corpus.stopwords.words('english'))
    return ' '.join([w for w in text.split() if w not in stop_words])


def delete_non_ascii(text: str) -> str:
    return ''.join([w for w in text if ord(w) < 128])


def replace_maj_word(text: str) -> str:
    token = '<MAJ>'
    return ' '.join([w if not w.isupper() else token for w in delete_punctuation(text).split()])


def delete_digit(text: str) -> str:
    return re.sub('[0-9]+', '', text)


def first_line(text: str) -> str:
    return re.split(r'[.!?]', text)[0]


def last_line(text: str) -> str:
    if text.endswith('\n'): text = text[:-2]
    return re.split(r'[.!?]', text)[-1]


def delete_balise(text: str) -> str:
    return re.sub("<.*?>", "", text)


def stem(text: str) -> str:
    stemmer = EnglishStemmer()
    tokens = nltk.word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    stemmed_text = " ".join(stemmed_tokens)
    return stemmed_text


def lemmatize(text: str) -> str:
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    lemmatized_text = " ".join(lemmatized_tokens)
    return lemmatized_text


def preprocess_text(text: str, args: Any, max_length: int=-1) -> str:
    text = str(text).strip()
    if args.replace_maj_word_flag: text = replace_maj_word(text)
    if args.lower_flag: text = text.lower()
    if args.delete_punctuation_flag: text = delete_punctuation(text)
    if args.delete_balise_flag: text = delete_balise(text)
    if args.delete_stopwords_flag: text = delete_stopwords(text)
    if args.delete_non_ascii_flag: text = delete_non_ascii(text)
    if args.delete_digit_flag: text = delete_digit(text)
    if args.first_line_flag: text = first_line(text)
    if args.last_line_flag: text = last_line(text)
    if args.stem_flag: text = stem(text)
    if args.lemmatize_flag: text = lemmatize(text)
    if max_length > 0 and args.truncate_flag:
        text = str(text).strip().split()
        if len(text) > max_length:
            text = text[:max_length - 1] + ["..."]
        text = " ".join(text)
    return text
