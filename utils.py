# Ben Kabongo
# January 2025

import matplotlib.pyplot as plt
import nltk
import numpy as np
import random
import re
import torch

from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import EnglishStemmer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def empty_cache():
    with torch.no_grad(): 
        torch.cuda.empty_cache()


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


def preprocess_text(config, text: str, max_length: int=-1) -> str:
    text = str(text).strip()
    if getattr(config, "replace_maj_word_flag", False): text = replace_maj_word(text)
    if getattr(config, "lower_flag", False): text = text.lower()
    if getattr(config, "delete_punctuation_flag", False): text = delete_punctuation(text)
    if getattr(config, "delete_balise_flag", False): text = delete_balise(text)
    if getattr(config, "delete_stopwords_flag", False): text = delete_stopwords(text)
    if getattr(config, "delete_non_ascii_flag", False): text = delete_non_ascii(text)
    if getattr(config, "delete_digit_flag", False): text = delete_digit(text)
    if getattr(config, "first_line_flag", False): text = first_line(text)
    if getattr(config, "last_line_flag", False): text = last_line(text)
    if getattr(config, "stem_flag", False): text = stem(text)
    if getattr(config, "lemmatize_flag", False): text = lemmatize(text)
    if max_length > 0:
        text = str(text).strip().split()
        if len(text) > max_length:
            text = text[:max_length]
        text = " ".join(text)
    return text


def plot_attention_weights(user_weights, item_weights, aspect_names, save_path=None):
    aspects = np.arange(len(aspect_names))
    data = np.stack([user_weights, item_weights], axis=1)
    fig, ax = plt.subplots(figsize=(4, 4))
    cax = ax.imshow(data, cmap='viridis', aspect='auto', interpolation='nearest')

    ax.set_yticks(aspects)
    ax.set_yticklabels(aspect_names, fontsize=12)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['User', 'Item'], fontsize=12)

    cbar = fig.colorbar(cax, ax=ax, orientation='vertical')
    cbar.set_label('Attention Intensity', fontsize=12)

    ax.grid(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
