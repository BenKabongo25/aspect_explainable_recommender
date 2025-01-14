# Ben Kabongo 
# November 2024

from utils import *


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

    def __init__(self, d):
        super().__init__()
        self.d = d
        self.WQ = nn.Linear(d, d)
        self.WK = nn.Linear(d, d)
        self.WV = nn.Linear(d, d)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                mask: torch.Tensor=None) -> Tuple[torch.Tensor]:
        Q = self.WQ(Q)
        K = self.WK(K)
        V = self.WV(V)
        attention_outputs, attention_scores = attention_function(Q, K, V, mask)
        return attention_outputs, attention_scores


class TextModule:

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

    def __init__(self, config, t5_model):
        super().__init__()
        self.config = config
        self.t5_model = t5_model

    def forward(self,
                tokens: torch.Tensor=None, mask: torch.Tensor=None,
                embeddings: torch.Tensor=None, labels: torch.Tensor=None,
                encode: bool=False, decode: bool=False) -> torch.Tensor:
        if encode and tokens is not None:
            return self.t5_model.encoder(input_ids=tokens, attention_mask=mask).last_hidden_state
        if decode and embeddings is not None:
            return self.t5_model(inputs_embeds=embeddings, labels=labels).loss

    def encode(self, tokens: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        return self.forward(tokens=tokens, mask=mask, encode=True)

    def decode(self, embeddings: torch.Tensor, labels: torch.Tensor=None) -> torch.Tensor:
        return self.forward(embeddings=embeddings, labels=labels, decode=True)

    def generate(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.t5_model.generate(
            inputs_embeds=embeddings,
            do_sample=False,
            max_length=self.config.review_length,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2
        )


class RatingsLoss(nn.Module):

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
        return {"total": total_loss, "overall_rating": overall_rating_loss, "aspect_rating": aspect_rating_loss}


class AURA(nn.Module):
    """ AURA: Aspect-based Unified Ratings Prediction and Personalized
        Review Generation with Attention """

    def __init__(self, config, text_module=None, tokenizer=None):
        super().__init__()
        self.config = config

        # Rating
        self.ratings_loss = RatingsLoss(config)

        self.user_embedding = nn.Embedding(config.n_users, config.d_model)
        self.item_embedding = nn.Embedding(config.n_items, config.d_model)
        self.user_attention = Attention(config.d_model)
        self.item_attention = Attention(config.d_model)

        self.user_aspects_embedding = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.ReLU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(config.d_model, config.d_model)
            ) for _ in range(config.n_aspects)
        ])
        self.item_aspects_embedding = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.ReLU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(config.d_model, config.d_model)
            ) for _ in range(config.n_aspects)
        ])

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

        # Review
        if config.review_flag:
            self.prompt_embedding = nn.Sequential(
                nn.Linear(2 * (1 + self.config.n_aspects) * config.d_model, config.n_prompt_tokens * config.d_words),
                nn.ReLU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(config.n_prompt_tokens * config.d_words, config.n_prompt_tokens * config.d_words)
            )
        self._init_weights()

        self.text_module = text_module
        self.tokenizer = tokenizer

        self.training_phase = 0 # 0: ratings, 1: prompt

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_training_phase(self, phase: int):
        self.training_phase = phase
        if phase == 0:
            self.ratings_grad(True)
            self.prompt_grad(False)
            self.text_grad(False)
        elif phase == 1:
            self.ratings_grad(False)
            self.prompt_grad(True)
            if not self.config.prompt_tuning:
                self.text_grad(False)

    def ratings_grad(self, flag: bool=True):
        for param in self.user_embedding.parameters():
            param.requires_grad = flag
        for param in self.item_embedding.parameters():
            param.requires_grad = flag
        for param in self.user_attention.parameters():
            param.requires_grad = flag
        for param in self.item_attention.parameters():
            param.requires_grad = flag
        for param in self.user_aspects_embedding.parameters():
            param.requires_grad = flag
        for param in self.item_aspects_embedding.parameters():
            param.requires_grad = flag
        for param in self.aspects_rating.parameters():
            param.requires_grad = flag
        for param in self.overall_rating.parameters():
            param.requires_grad = flag

    def prompt_grad(self, flag: bool=True):
        if not self.config.review_flag:
            return
        for param in self.prompt_embedding.parameters():
            param.requires_grad = flag

    def text_grad(self, flag: bool=True):
        if not self.config.review_flag:
            return
        for param in self.text_module.parameters():
            param.requires_grad = flag

    def forward(self, U_ids: torch.Tensor, I_ids: torch.Tensor,
                R: torch.Tensor=None, A_ratings: torch.Tensor=None,
                review_tokens: torch.Tensor=None, inference_flag: bool=False) -> Dict[str, torch.Tensor]:
        # U_ids, I_ids: (batch_size,)
        # review_tokens: (batch_size, seq_len)

        U_embeddings = self.user_embedding(U_ids) # (batch_size, d_model)
        I_embeddings = self.item_embedding(I_ids) # (batch_size, d_model)

        _out = {}

        UA_embeddings = []
        IA_embeddings = []
        A_ratings_hat = []
        for i in range(self.config.n_aspects):
            au_embedding = self.user_aspects_embedding[i](U_embeddings) # (batch_size, d_model)
            ai_embedding = self.item_aspects_embedding[i](I_embeddings) # (batch_size, d_model)
            UA_embeddings.append(au_embedding)
            IA_embeddings.append(ai_embedding)

            if inference_flag or self.training_phase == 0:
                a_rating = self.aspects_rating[i](torch.cat([au_embedding, ai_embedding], dim=-1)) # (batch_size,)
                #a_rating = torch.clamp(a_rating, min=self.config.min_rating, max=self.config.max_rating)
                A_ratings_hat.append(a_rating)

        UA_embeddings = torch.stack(UA_embeddings, dim=1) # (batch_size, n_aspects, d_model)
        IA_embeddings = torch.stack(IA_embeddings, dim=1) # (batch_size, n_aspects, d_model)
            
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

        _out.update({
            "user": {
                "attention": U_attention_scores,
            },
            "item": {
                "attention": I_attention_scores,
            }
        })

        if not self.config.review_flag or inference_flag or self.training_phase == 0:
            A_ratings_hat = torch.stack(A_ratings_hat, dim=1).squeeze(2) # (batch_size, n_aspects)
            R_hat = self.overall_rating(torch.cat([U_embeddings_aggregated, I_embeddings_aggregated], dim=-1)).squeeze(1)
            #R_hat = torch.clamp(R_hat, min=self.config.min_rating, max=self.config.max_rating)
            losses = self.ratings_loss(R, R_hat, A_ratings, A_ratings_hat)
            _out.update({
                "overall_rating": R_hat,
                "aspects_ratings": A_ratings_hat,
                "losses": losses
            })

        if self.config.review_flag and (inference_flag or self.training_phase == 1):
            P_embeddings = torch.cat(
                [U_embeddings.unsqueeze(1), I_embeddings.unsqueeze(1), IA_embeddings, IA_embeddings], 
                dim=1
            ).view(U_embeddings.size(0), -1) # (batch_size, 2 * (1 + n_aspects) * d_model)
            P_embeddings = self.prompt_embedding(P_embeddings).view(U_embeddings.size(0), -1, self.config.d_words) # (batch_size, n_prompt_tokens, d_words)
            
            if not inference_flag:
                review_loss = self.text_module.decode(P_embeddings, review_tokens)
                losses = {"review": review_loss, "total": review_loss}
                _out.update({"losses": losses})

            else:
                review_tokens_hat = self.text_module.generate(P_embeddings)
                review = self.tokenizer.batch_decode(review_tokens_hat, skip_special_tokens=True)
                _out.update({'review': review})

        return _out
    