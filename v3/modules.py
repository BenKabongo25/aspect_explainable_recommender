# Ben Kabongo
# January 2025

# AURA: Aspect-based Unified Ratings Prediction and 
# Personalized Review Generation with Attention


import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple, Dict


def _get_module(input_dim, hidden_dim, output_dim, n_layers, dropout):
    layers = []
    layers.append(nn.Linear(input_dim, hidden_dim))
    layers.append(nn.ReLU())
    layers.append(nn.Dropout(dropout))
    for _ in range(n_layers):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden_dim, output_dim))
    module = nn.Sequential(*layers)
    return module


class ModuleOutput:
    """ Module output """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def attention_function(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                       mask: torch.Tensor=None) -> Tuple[torch.Tensor]:
    attention_scores = F.softmax(
        torch.matmul(Q, K.transpose(1, 2)) / torch.sqrt(torch.tensor(Q.size(-1))),
        dim=-1
    )
    if mask is not None:
        mask = mask.unsqueeze(1) if mask.dim() == 2 else mask
        attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
    attention_outputs = torch.matmul(attention_scores, V)
    return attention_outputs, attention_scores


class Attention(nn.Module):
    """ Attention module """

    def __init__(self, d):
        super().__init__()
        self.d = d
        self.WQ = nn.Linear(d, d, bias=False)
        self.WK = nn.Linear(d, d, bias=False)
        self.WV = nn.Linear(d, d, bias=False)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                mask: torch.Tensor=None) -> Tuple[torch.Tensor]:
        Q = self.WQ(Q)
        K = self.WK(K)
        V = self.WV(V)
        attention_outputs, attention_scores = attention_function(Q, K, V, mask)
        return attention_outputs, attention_scores


class AspectsRatingLoss(nn.Module):
    """ Aspects + overall rating loss """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.overall_rating_loss = nn.MSELoss()
        self.aspect_rating_loss = nn.MSELoss()

    def forward(self, R: torch.Tensor, R_hat: torch.Tensor,
                A_ratings: torch.Tensor, A_ratings_hat: torch.Tensor) -> Dict[str, torch.Tensor]:
        overall_rating_loss = self.overall_rating_loss(R_hat, R)
        aspect_rating_loss = self.aspect_rating_loss(A_ratings_hat.flatten(), A_ratings.flatten())
        total_loss = self.config.alpha * overall_rating_loss + self.config.beta * aspect_rating_loss
        _out = {"total": total_loss, "overall_rating": overall_rating_loss, "aspects_ratings": aspect_rating_loss}
        return ModuleOutput(**_out)


class RatingPredictionModule(nn.Module):
    """ Rating prediction module without aspects """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.user_embedding = nn.Embedding(config.n_users, config.d_model)
        self.item_embedding = nn.Embedding(config.n_items, config.d_model)
        self.overall_rating = _get_module(
            config.d_model * 2, config.d_model, 1, config.n_layers, config.dropout
        )

    def forward(self, U_ids: torch.Tensor, I_ids: torch.Tensor) -> torch.Tensor:
        U_embeddings = self.user_embedding(U_ids) # (batch_size, d_model)
        I_embeddings = self.item_embedding(I_ids) # (batch_size, d_model)
        R = self.overall_rating(torch.cat([U_embeddings, I_embeddings], dim=-1)).squeeze(-1)
        R = torch.clamp(R, min=self.config.min_rating, max=self.config.max_rating)
        _out = {
            "U_embeddings": U_embeddings,
            "I_embeddings": I_embeddings,
            "overall_rating": R
        }
        return ModuleOutput(**_out)
    

class AspectsRatingPredictionModule(nn.Module):
    """ Rating prediction module with aspects """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.user_embedding = nn.Embedding(config.n_users, config.d_model)
        self.item_embedding = nn.Embedding(config.n_items, config.d_model)
        self.overall_rating = _get_module(
            config.d_model * 2, config.d_model, 1, config.n_layers, config.dropout
        )
        self.user_attention = Attention(config.d_model)
        self.user_aspects_embedding = nn.ModuleList([
            _get_module(config.d_model * 2, config.d_model, config.d_model, config.n_layers, config.dropout)
            for _ in range(config.n_aspects)
        ])
        self.item_attention = Attention(config.d_model)
        self.item_aspects_embedding = nn.ModuleList([
            _get_module(config.d_model * 2, config.d_model, config.d_model, config.n_layers, config.dropout)
            for _ in range(config.n_aspects)
        ])
        self.aspects_rating = nn.ModuleList([
            _get_module(config.d_model * 2, config.d_model, 1, config.n_layers, config.dropout)
            for _ in range(config.n_aspects)
        ])

    def forward(self, U_ids: torch.Tensor, I_ids: torch.Tensor) -> torch.Tensor:
        U_embeddings = self.user_embedding(U_ids) # (batch_size, d_model)
        I_embeddings = self.item_embedding(I_ids) # (batch_size, d_model)

        UA_embeddings = []
        IA_embeddings = []
        A_ratings_hat = []
        for i in range(self.config.n_aspects):
            au_embeddings = self.user_aspects_embedding[i](U_embeddings) # (batch_size, d_model)
            ai_embeddings = self.item_aspects_embedding[i](I_embeddings) # (batch_size, d_model)
            UA_embeddings.append(au_embeddings)
            IA_embeddings.append(ai_embeddings)

            a_rating = self.aspects_rating[i](torch.cat([au_embeddings, ai_embeddings], dim=-1)) # (batch_size,)
            a_rating = torch.clamp(a_rating, min=self.config.min_rating, max=self.config.max_rating)
            A_ratings_hat.append(a_rating)

        UA_embeddings = torch.stack(UA_embeddings, dim=1) # (batch_size, n_aspects, d_model)
        IA_embeddings = torch.stack(IA_embeddings, dim=1) # (batch_size, n_aspects, d_model)
        A_ratings_hat = torch.stack(A_ratings_hat, dim=1).squeeze(2) # (batch_size, n_aspects)
            
        U_embeddings_aggregated, U_attention_scores = self.user_attention(
            Q=U_embeddings.unsqueeze(1), K=UA_embeddings, V=UA_embeddings
        ) # self attention
        U_attention_scores = U_attention_scores.squeeze(1) # (batch_size, n_aspects)
        U_embeddings_aggregated = U_embeddings_aggregated.squeeze(1) # (batch_size, d_model)

        I_embeddings_aggregated, I_attention_scores = self.item_attention(
            Q=I_embeddings.unsqueeze(1), K=IA_embeddings, V=IA_embeddings
        ) # self attention
        I_attention_scores = I_attention_scores.squeeze(1) # (batch_size, n_aspects)
        I_embeddings_aggregated = I_embeddings_aggregated.squeeze(1) # (batch_size, d_model)

        R_hat = self.overall_rating(torch.cat([U_embeddings_aggregated, I_embeddings_aggregated], dim=-1)).squeeze(1) # (batch_size,)
        R_hat = torch.clamp(R_hat, min=self.config.min_rating, max=self.config.max_rating)

        _out = {
            "U_embeddings": U_embeddings,
            "I_embeddings": I_embeddings,
            "UA_embeddings": UA_embeddings,
            "IA_embeddings": IA_embeddings,
            "U_embeddings_aggregated": U_embeddings_aggregated,
            "I_embeddings_aggregated": I_embeddings_aggregated,
            "U_attention_scores": U_attention_scores,
            "I_attention_scores": I_attention_scores,
            "A_ratings_hat": A_ratings_hat,
            "overall_rating": R_hat
        }
        return ModuleOutput(**_out)
    

class TextModule:
    """ Text module """

    def encode(self, tokens: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        # tokens: (batch_size, seq_len)
        # mask: (batch_size, seq_len)
        raise NotImplementedError

    def decode(self, embeddings: torch.Tensor, labels: torch.Tensor=None) -> torch.Tensor:
        # embeddings: (batch_size, seq_len, d_words)
        # labels: (batch_size, seq_len)
        raise NotImplementedError

    def generate(self, embeddings: torch.Tensor) -> List[str]:
        # embeddings: (batch_size, seq_len, d_words)
        raise NotImplementedError


class T5TextModule(nn.Module, TextModule):
    """ T5 text module """

    def __init__(self, config, t5_model):
        super().__init__()
        self.config = config
        self.t5_model = t5_model

    def forward(self, embeddings: torch.Tensor=None, labels: torch.Tensor=None) -> torch.Tensor:
        return self.t5_model(inputs_embeds=embeddings, labels=labels).loss

    def encode(self, tokens: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        return self.forward(tokens=tokens, mask=mask, encode=True)

    def decode(self, embeddings: torch.Tensor, labels: torch.Tensor=None) -> torch.Tensor:
        return self.forward(embeddings=embeddings, labels=labels)

    def generate(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.t5_model.generate(
            inputs_embeds=embeddings,
            do_sample=False,
            max_length=self.config.review_length,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2
        )


class ReviewGenerationModule(nn.Module):
    """ Personalized review generation module """

    def __init__(self, config, text_module, tokenizer):
        super().__init__()
        self.config = config
        self.text_module = text_module
        self.tokenizer = tokenizer

        self.input_dim = config.n_prompt_elements * config.d_model
        self.output_dim = config.n_prompt_tokens * config.d_words
        self.prompt_embedding = _get_module(
            self.input_dim, self.output_dim, self.output_dim, config.n_layers, config.dropout
        )

    def prompt(self, U_embeddings: torch.Tensor, I_embeddings: torch.Tensor,
                UA_embeddings: torch.Tensor=None, IA_embeddings: torch.Tensor=None) -> torch.Tensor:
        P_embeddings = [U_embeddings.unsqueeze(1), I_embeddings.unsqueeze(1)]
        if UA_embeddings is not None and IA_embeddings is not None:
            P_embeddings.extend([UA_embeddings, IA_embeddings])

        P_embeddings = torch.cat(P_embeddings, dim=1) # (batch_size, n_prompt_elements, d_model)
        P_embeddings = P_embeddings.view(P_embeddings.size(0), -1) # (batch_size, n_prompt_elements * d_model)
        P_embeddings = self.prompt_embedding(P_embeddings) # (batch_size, n_prompt_tokens * d_words)
        P_embeddings = P_embeddings.view(P_embeddings.size(0), -1, self.config.d_words) # (batch_size, n_prompt_tokens, d_words)
        return P_embeddings

    def forward(self, U_embeddings: torch.Tensor, I_embeddings: torch.Tensor, labels: torch.Tensor,
                UA_embeddings: torch.Tensor=None, IA_embeddings: torch.Tensor=None) -> torch.Tensor:
        P_embeddings = self.prompt(U_embeddings, I_embeddings, UA_embeddings, IA_embeddings)
        loss = self.text_module.decode(P_embeddings, labels).loss
        _out = {"loss": loss, "P_embeddings": P_embeddings}
        return ModuleOutput(**_out)
    
    def generate(self, U_embeddings: torch.Tensor, I_embeddings: torch.Tensor,
                 UA_embeddings: torch.Tensor=None, IA_embeddings: torch.Tensor=None) -> List[str]:
        P_embeddings = self.prompt(U_embeddings, I_embeddings, UA_embeddings, IA_embeddings)
        return self.text_module.generate(P_embeddings)


class AURA(nn.Module):
    """ AURA for joint training """

    def __init__(self, config, rating_module, review_module):
        super().__init__()
        self.config = config
        self.rating_module = rating_module
        self.review_module = review_module

    def forward(self, U_ids: torch.Tensor, I_ids: torch.Tensor,
                labels: torch.Tensor=None) -> Dict[str, torch.Tensor]:
        rating_output = self.rating_module(U_ids, I_ids)
        review_output = self.review_module(
            rating_output.U_embeddings, rating_output.I_embeddings, labels=labels
        )
        return {
            "ratings": rating_output,
            "reviews": review_output
        }
    
    def generate(self, U_ids: torch.Tensor, I_ids: torch.Tensor) -> List[str]:
        rating_output = self.rating_module(U_ids, I_ids)
        return self.review_module.generate(rating_output.U_embeddings, rating_output.I_embeddings)
    