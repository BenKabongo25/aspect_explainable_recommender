# Ben Kabongo
# December 2024

from main import *


datasets = ["Hotels", "Beer"]
aspects = {"Hotels": ["cleanliness", "location", "rooms", "service", "sleep_quality", "value"],
           "Beer": ["appearance", "aroma", "palate", "taste"]}

class Config:
    pass

config = Config()
config.lang = "en"
config.review_flag = False
config.rating_flag = True
config.model_name_or_path = "t5-small"
config.d_words = 512
config.d_model = 128
config.review_length = 256
config.n_ratings_epochs = 50
config.n_reviews_epochs = 0
config.dropout = 0.1
config.rating_metric = "rmse"
config.review_metric = "meteor"
config.lr = 1e-3
config.batch_size = 512
config.train_size = 0.8
config.eval_size = 0.1
config.test_size = 0.1
config.seed = 42
config.load_model = False
config.min_rating = 1.0
config.max_rating = 5.0
config.threshold_rating = 4.0
config.k = 10
config.ranking_metrics_flag = False
config.verbose = True
config.verbose_every = 1
config.eval_every = 1
config.truncate_flag = True
config.lower_flag = True
config.delete_balise_flag = True
config.delete_stopwords_flag = False
config.delete_punctuation_flag = False
config.delete_non_ascii_flag = True
config.delete_digit_flag = False
config.replace_maj_word_flag = False
config.first_line_flag = False
config.last_line_flag = False
config.stem_flag = False
config.lemmatize_flag = False

for dropout in [0.1, 0.2, 0.3, 0.4, 0.5]:
    config.dropout = dropout

    for architecture in [""]:
        config.architecture = architecture
        architecture = "AURA" + architecture

        for dataset in datasets:
            config.dataset = dataset
            config.aspects = aspects[dataset]
            config.dataset_name = dataset
            config.alpha = 1 / (1 + len(config.aspects))
            config.beta = len(config.aspects) / (1 + len(config.aspects))

            config.dataset_path = os.path.join("/home", "b.kabongo", "aspects_datasets", dataset, "data.csv")
            config.save_dir = os.path.join("/home", "b.kabongo", "exps256", dataset, architecture + f"_dropout_{dropout}")

            print(f"Running {dataset} with {architecture} and dropout {dropout}")
            run(config)
