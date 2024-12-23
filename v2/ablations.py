# Ben Kabongo
# December 2024

from module import *


class AURAPooling(nn.Module):
    """ AURA with pooling instead of attention
    """

    def __init__(self, config, text_module=None, tokenizer=None):
        super().__init__()
        self.config = config

        # Rating
        self.ratings_loss = RatingsLoss(config)

        self.user_embedding = nn.Embedding(config.n_users, config.d_model)
        self.item_embedding = nn.Embedding(config.n_items, config.d_model)

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
            self.text_module = text_module
            self.tokenizer = tokenizer

            self.user_words_proj = nn.Linear(config.d_model, config.d_words)
            self.item_words_proj = nn.Linear(config.d_model, config.d_words)
            self.user_aspect_words_proj = nn.Linear(config.d_model, config.d_words)
            self.item_aspect_words_proj = nn.Linear(config.d_model, config.d_words)
            #self.prompt_embedding = nn.Sequential(
            #    nn.Linear(2 * (1 + self.config.n_aspects) * config.d_words, config.n_prompt * config.d_words),
            #    nn.ReLU(),
            #    nn.Linear(config.n_prompt * config.d_words, config.n_prompt * config.d_words)
            #)

        self.training_phase = 0 # 0: ratings, 1: prompt, 2: text module + prompt
        self._init_weights()

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
            self.prompt_grad(True)
            self.ratings_grad(False)
            self.text_grad(False)
        elif phase == 2:
            self.text_grad(True)

    def ratings_grad(self, flag: bool=True):
        for param in self.user_embedding.parameters():
            param.requires_grad = flag
        for param in self.item_embedding.parameters():
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
        for param in self.user_words_proj.parameters():
            param.requires_grad = flag
        for param in self.item_words_proj.parameters():
            param.requires_grad = flag
        for param in self.user_aspect_words_proj.parameters():
            param.requires_grad = flag
        for param in self.item_aspect_words_proj.parameters():
            param.requires_grad = flag
        #for param in self.prompt_embedding.parameters():
        #    param.requires_grad = flag

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
            
        U_embeddings_aggregated = UA_embeddings.max(dim=1).values # (batch_size, d_model)
        U_embeddings = U_embeddings_aggregated # no skip connection

        I_embeddings_aggregated = IA_embeddings.max(dim=1).values # (batch_size, d_model)
        I_embeddings = I_embeddings_aggregated # no skip connection

        if not self.config.review_flag or inference_flag or self.training_phase == 0:
            A_ratings_hat = torch.stack(A_ratings_hat, dim=1).squeeze(2) # (batch_size, n_aspects)
            R_hat = self.overall_rating(torch.cat([U_embeddings, I_embeddings], dim=-1)).squeeze(1)
            #R_hat = torch.clamp(R_hat, min=self.config.min_rating, max=self.config.max_rating)
            losses = self.ratings_loss(R, R_hat, A_ratings, A_ratings_hat)
            _out.update({
                "overall_rating": R_hat,
                "aspects_ratings": A_ratings_hat,
                "losses": losses
            })

        if self.config.review_flag and (inference_flag or self.training_phase == 1):
            U_embeddings_words = self.user_words_proj(U_embeddings).unsqueeze(1) # (batch_size, 1, d_words)
            I_embeddings_words = self.item_words_proj(I_embeddings).unsqueeze(1) # (batch_size, 1, d_words)
            UA_embeddings_words = self.user_aspect_words_proj(UA_embeddings) # (batch_size, n_aspects, d_words)
            IA_embeddings_words = self.item_aspect_words_proj(IA_embeddings) # (batch_size, n_aspects, d_words)
            P_embeddings = torch.cat(
                [U_embeddings_words, I_embeddings_words, UA_embeddings_words, IA_embeddings_words], 
                dim=1
            ) # (batch_size, 2 * (1 + n_aspects), d_words)

            if not inference_flag:
                review_loss = self.text_module.decode(P_embeddings, review_tokens)
                losses = {"review": review_loss, "total": review_loss}
                _out.update({"losses": losses})

            else:
                review_tokens_hat = self.text_module.generate(P_embeddings)
                review = self.tokenizer.batch_decode(review_tokens_hat, skip_special_tokens=True)
                _out.update({'review': review})

        return _out


class AURAWithSkipConnection(nn.Module):
    """ AURA with skip connection
    """

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
            self.text_module = text_module
            self.tokenizer = tokenizer

            self.user_words_proj = nn.Linear(config.d_model, config.d_words)
            self.item_words_proj = nn.Linear(config.d_model, config.d_words)
            self.user_aspect_words_proj = nn.Linear(config.d_model, config.d_words)
            self.item_aspect_words_proj = nn.Linear(config.d_model, config.d_words)
            #self.prompt_embedding = nn.Sequential(
            #    nn.Linear(2 * (1 + self.config.n_aspects) * config.d_words, config.n_prompt * config.d_words),
            #    nn.ReLU(),
            #    nn.Linear(config.n_prompt * config.d_words, config.n_prompt * config.d_words)
            #)

        self.training_phase = 0 # 0: ratings, 1: prompt, 2: text module + prompt
        self._init_weights()

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
            self.prompt_grad(True)
            self.ratings_grad(False)
            self.text_grad(False)
        elif phase == 2:
            self.text_grad(True)

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
        for param in self.user_words_proj.parameters():
            param.requires_grad = flag
        for param in self.item_words_proj.parameters():
            param.requires_grad = flag
        for param in self.user_aspect_words_proj.parameters():
            param.requires_grad = flag
        for param in self.item_aspect_words_proj.parameters():
            param.requires_grad = flag
        #for param in self.prompt_embedding.parameters():
        #    param.requires_grad = flag

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
        U_embeddings = U_embeddings + U_embeddings_aggregated # skip connection

        I_embeddings_aggregated, I_attention_scores = self.item_attention(
            Q=I_embeddings.unsqueeze(1), K=IA_embeddings, V=IA_embeddings
        ) # self attention
        I_attention_scores = I_attention_scores.squeeze(1) # (batch_size, n_aspects)
        I_embeddings_aggregated = I_embeddings_aggregated.squeeze(1) # (batch_size, d_model)
        I_embeddings = I_embeddings + I_embeddings_aggregated # skip connection

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
            R_hat = self.overall_rating(torch.cat([U_embeddings, I_embeddings], dim=-1)).squeeze(1)
            #R_hat = torch.clamp(R_hat, min=self.config.min_rating, max=self.config.max_rating)
            losses = self.ratings_loss(R, R_hat, A_ratings, A_ratings_hat)
            _out.update({
                "overall_rating": R_hat,
                "aspects_ratings": A_ratings_hat,
                "losses": losses
            })


        if self.config.review_flag and (inference_flag or self.training_phase == 1):
            U_embeddings_words = self.user_words_proj(U_embeddings).unsqueeze(1) # (batch_size, 1, d_words)
            I_embeddings_words = self.item_words_proj(I_embeddings).unsqueeze(1) # (batch_size, 1, d_words)
            UA_embeddings_words = self.user_aspect_words_proj(UA_embeddings) # (batch_size, n_aspects, d_words)
            IA_embeddings_words = self.item_aspect_words_proj(IA_embeddings) # (batch_size, n_aspects, d_words)
            P_embeddings = torch.cat(
                [U_embeddings_words, I_embeddings_words, UA_embeddings_words, IA_embeddings_words], 
                dim=1
            ) # (batch_size, 2 * (1 + n_aspects), d_words)

            if not inference_flag:
                review_loss = self.text_module.decode(P_embeddings, review_tokens)
                losses = {"review": review_loss, "total": review_loss}
                _out.update({"losses": losses})

            else:
                review_tokens_hat = self.text_module.generate(P_embeddings)
                review = self.tokenizer.batch_decode(review_tokens_hat, skip_special_tokens=True)
                _out.update({'review': review})

        return _out


class AURASharedAspectsEmbedding(nn.Module):
    """ AURA with shared aspects embedding
    """

    def __init__(self, config, text_module=None, tokenizer=None):
        super().__init__()
        self.config = config

        # Rating
        self.ratings_loss = RatingsLoss(config)

        self.user_embedding = nn.Embedding(config.n_users, config.d_model)
        self.item_embedding = nn.Embedding(config.n_items, config.d_model)
        self.user_attention = Attention(config.d_model)
        self.item_attention = Attention(config.d_model)

        self.aspects_embedding = nn.ModuleList([
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
            self.text_module = text_module
            self.tokenizer = tokenizer

            self.user_words_proj = nn.Linear(config.d_model, config.d_words)
            self.item_words_proj = nn.Linear(config.d_model, config.d_words)
            self.user_aspect_words_proj = nn.Linear(config.d_model, config.d_words)
            self.item_aspect_words_proj = nn.Linear(config.d_model, config.d_words)
            #self.prompt_embedding = nn.Sequential(
            #    nn.Linear(2 * (1 + self.config.n_aspects) * config.d_words, config.n_prompt * config.d_words),
            #    nn.ReLU(),
            #    nn.Linear(config.n_prompt * config.d_words, config.n_prompt * config.d_words)
            #)

        self.training_phase = 0 # 0: ratings, 1: prompt, 2: text module + prompt
        self._init_weights()

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
            self.prompt_grad(True)
            self.ratings_grad(False)
            self.text_grad(False)
        elif phase == 2:
            self.text_grad(True)

    def ratings_grad(self, flag: bool=True):
        for param in self.user_embedding.parameters():
            param.requires_grad = flag
        for param in self.item_embedding.parameters():
            param.requires_grad = flag
        for param in self.user_attention.parameters():
            param.requires_grad = flag
        for param in self.item_attention.parameters():
            param.requires_grad = flag
        for param in self.aspects_rating.parameters():
            param.requires_grad = flag
        for param in self.overall_rating.parameters():
            param.requires_grad = flag

    def prompt_grad(self, flag: bool=True):
        if not self.config.review_flag:
            return
        for param in self.user_words_proj.parameters():
            param.requires_grad = flag
        for param in self.item_words_proj.parameters():
            param.requires_grad = flag
        for param in self.user_aspect_words_proj.parameters():
            param.requires_grad = flag
        for param in self.item_aspect_words_proj.parameters():
            param.requires_grad = flag
        #for param in self.prompt_embedding.parameters():
        #    param.requires_grad = flag

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
            au_embedding = self.aspects_embedding[i](U_embeddings) # (batch_size, d_model)
            ai_embedding = self.aspects_embedding[i](I_embeddings) # (batch_size, d_model)
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
        )
        U_attention_scores = U_attention_scores.squeeze(1) # (batch_size, n_aspects)
        U_embeddings_aggregated = U_embeddings_aggregated.squeeze(1) # (batch_size, d_model)
        U_embeddings = U_embeddings_aggregated # no skip connection

        I_embeddings_aggregated, I_attention_scores = self.item_attention(
            Q=I_embeddings.unsqueeze(1), K=IA_embeddings, V=IA_embeddings
        )
        I_attention_scores = I_attention_scores.squeeze(1) # (batch_size, n_aspects)
        I_embeddings_aggregated = I_embeddings_aggregated.squeeze(1) # (batch_size, d_model)
        I_embeddings = I_embeddings_aggregated # no skip connection

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
            R_hat = self.overall_rating(torch.cat([U_embeddings, I_embeddings], dim=-1)).squeeze(1)
            #R_hat = torch.clamp(R_hat, min=self.config.min_rating, max=self.config.max_rating)
            losses = self.ratings_loss(R, R_hat, A_ratings, A_ratings_hat)
            _out.update({
                "overall_rating": R_hat,
                "aspects_ratings": A_ratings_hat,
                "losses": losses
            })

        if self.config.review_flag and (inference_flag or self.training_phase == 1):
            U_embeddings_words = self.user_words_proj(U_embeddings).unsqueeze(1) # (batch_size, 1, d_words)
            I_embeddings_words = self.item_words_proj(I_embeddings).unsqueeze(1) # (batch_size, 1, d_words)
            UA_embeddings_words = self.user_aspect_words_proj(UA_embeddings) # (batch_size, n_aspects, d_words)
            IA_embeddings_words = self.item_aspect_words_proj(IA_embeddings) # (batch_size, n_aspects, d_words)
            P_embeddings = torch.cat(
                [U_embeddings_words, I_embeddings_words, UA_embeddings_words, IA_embeddings_words], 
                dim=1
            ) # (batch_size, 2 * (1 + n_aspects), d_words)

            if not inference_flag:
                review_loss = self.text_module.decode(P_embeddings, review_tokens)
                losses = {"review": review_loss, "total": review_loss}
                _out.update({"losses": losses})

            else:
                review_tokens_hat = self.text_module.generate(P_embeddings)
                review = self.tokenizer.batch_decode(review_tokens_hat, skip_special_tokens=True)
                _out.update({'review': review})

        return _out
    

class AURASharedAspectsRating(nn.Module):

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

        self.aspect_rating = nn.Sequential(
            nn.Linear(2 * config.d_model, config.d_model),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(config.d_model, 1)
        )
        self.overall_rating = nn.Sequential(
            nn.Linear(2 * config.d_model, config.d_model),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(config.d_model, 1)
        )

        # Review
        if config.review_flag:
            self.text_module = text_module
            self.tokenizer = tokenizer

            self.user_words_proj = nn.Linear(config.d_model, config.d_words)
            self.item_words_proj = nn.Linear(config.d_model, config.d_words)
            self.user_aspect_words_proj = nn.Linear(config.d_model, config.d_words)
            self.item_aspect_words_proj = nn.Linear(config.d_model, config.d_words)
            #self.prompt_embedding = nn.Sequential(
            #    nn.Linear(2 * (1 + self.config.n_aspects) * config.d_words, config.n_prompt * config.d_words),
            #    nn.ReLU(),
            #    nn.Linear(config.n_prompt * config.d_words, config.n_prompt * config.d_words)
            #)

        self.training_phase = 0 # 0: ratings, 1: prompt, 2: text module + prompt
        self._init_weights()

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
            self.prompt_grad(True)
            self.ratings_grad(False)
            self.text_grad(False)
        elif phase == 2:
            self.text_grad(True)

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
        for param in self.overall_rating.parameters():
            param.requires_grad = flag
        for param in self.aspect_rating.parameters():
            param.requires_grad = flag

    def prompt_grad(self, flag: bool=True):
        if not self.config.review_flag:
            return
        for param in self.user_words_proj.parameters():
            param.requires_grad = flag
        for param in self.item_words_proj.parameters():
            param.requires_grad = flag
        for param in self.user_aspect_words_proj.parameters():
            param.requires_grad = flag
        for param in self.item_aspect_words_proj.parameters():
            param.requires_grad = flag
        #for param in self.prompt_embedding.parameters():
        #    param.requires_grad = flag

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
                a_rating = self.aspect_rating(torch.cat([au_embedding, ai_embedding], dim=-1)) # (batch_size,)
                #a_rating = torch.clamp(a_rating, min=self.config.min_rating, max=self.config.max_rating)
                A_ratings_hat.append(a_rating)

        UA_embeddings = torch.stack(UA_embeddings, dim=1) # (batch_size, n_aspects, d_model)
        IA_embeddings = torch.stack(IA_embeddings, dim=1) # (batch_size, n_aspects, d_model)
            
        U_embeddings_aggregated, U_attention_scores = self.user_attention(
            Q=U_embeddings.unsqueeze(1), K=UA_embeddings, V=UA_embeddings
        ) # self attention
        U_attention_scores = U_attention_scores.squeeze(1) # (batch_size, n_aspects)
        U_embeddings_aggregated = U_embeddings_aggregated.squeeze(1) # (batch_size, d_model)
        U_embeddings = U_embeddings_aggregated # no skip connection

        I_embeddings_aggregated, I_attention_scores = self.item_attention(
            Q=I_embeddings.unsqueeze(1), K=IA_embeddings, V=IA_embeddings
        ) # self attention
        I_attention_scores = I_attention_scores.squeeze(1) # (batch_size, n_aspects)
        I_embeddings_aggregated = I_embeddings_aggregated.squeeze(1) # (batch_size, d_model)
        I_embeddings = I_embeddings_aggregated # no skip connection

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
            R_hat = self.overall_rating(torch.cat([U_embeddings, I_embeddings], dim=-1)).squeeze(1)
            #R_hat = torch.clamp(R_hat, min=self.config.min_rating, max=self.config.max_rating)
            losses = self.ratings_loss(R, R_hat, A_ratings, A_ratings_hat)
            _out.update({
                "overall_rating": R_hat,
                "aspects_ratings": A_ratings_hat,
                "losses": losses
            })

        if self.config.review_flag and (inference_flag or self.training_phase == 1):
            U_embeddings_words = self.user_words_proj(U_embeddings).unsqueeze(1) # (batch_size, 1, d_words)
            I_embeddings_words = self.item_words_proj(I_embeddings).unsqueeze(1) # (batch_size, 1, d_words)
            UA_embeddings_words = self.user_aspect_words_proj(UA_embeddings) # (batch_size, n_aspects, d_words)
            IA_embeddings_words = self.item_aspect_words_proj(IA_embeddings) # (batch_size, n_aspects, d_words)
            P_embeddings = torch.cat(
                [U_embeddings_words, I_embeddings_words, UA_embeddings_words, IA_embeddings_words], 
                dim=1
            ) # (batch_size, 2 * (1 + n_aspects), d_words)

            if not inference_flag:
                review_loss = self.text_module.decode(P_embeddings, review_tokens)
                losses = {"review": review_loss, "total": review_loss}
                _out.update({"losses": losses})

            else:
                review_tokens_hat = self.text_module.generate(P_embeddings)
                review = self.tokenizer.batch_decode(review_tokens_hat, skip_special_tokens=True)
                _out.update({'review': review})

        return _out


class AURASharedAggregation(nn.Module):
    """ AURA with shared aggregation
    """

    def __init__(self, config, text_module=None, tokenizer=None):
        super().__init__()
        self.config = config

        # Rating
        self.ratings_loss = RatingsLoss(config)

        self.user_embedding = nn.Embedding(config.n_users, config.d_model)
        self.item_embedding = nn.Embedding(config.n_items, config.d_model)
        self.attention = Attention(config.d_model)

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
            self.text_module = text_module
            self.tokenizer = tokenizer

            self.user_words_proj = nn.Linear(config.d_model, config.d_words)
            self.item_words_proj = nn.Linear(config.d_model, config.d_words)
            self.user_aspect_words_proj = nn.Linear(config.d_model, config.d_words)
            self.item_aspect_words_proj = nn.Linear(config.d_model, config.d_words)
            #self.prompt_embedding = nn.Sequential(
            #    nn.Linear(2 * (1 + self.config.n_aspects) * config.d_words, config.n_prompt * config.d_words),
            #    nn.ReLU(),
            #    nn.Linear(config.n_prompt * config.d_words, config.n_prompt * config.d_words)
            #)

        self.training_phase = 0 # 0: ratings, 1: prompt, 2: text module + prompt
        self._init_weights()

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
            self.prompt_grad(True)
            self.ratings_grad(False)
            self.text_grad(False)
        elif phase == 2:
            self.text_grad(True)

    def ratings_grad(self, flag: bool=True):
        for param in self.user_embedding.parameters():
            param.requires_grad = flag
        for param in self.item_embedding.parameters():
            param.requires_grad = flag
        for param in self.attention.parameters():
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
        for param in self.user_words_proj.parameters():
            param.requires_grad = flag
        for param in self.item_words_proj.parameters():
            param.requires_grad = flag
        for param in self.user_aspect_words_proj.parameters():
            param.requires_grad = flag
        for param in self.item_aspect_words_proj.parameters():
            param.requires_grad = flag
        #for param in self.prompt_embedding.parameters():
        #    param.requires_grad = flag

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
            
        U_embeddings_aggregated, U_attention_scores = self.attention(
            Q=U_embeddings.unsqueeze(1), K=UA_embeddings, V=UA_embeddings
        ) # self attention
        U_attention_scores = U_attention_scores.squeeze(1) # (batch_size, n_aspects)
        U_embeddings_aggregated = U_embeddings_aggregated.squeeze(1) # (batch_size, d_model)
        U_embeddings = U_embeddings_aggregated # no skip connection

        I_embeddings_aggregated, I_attention_scores = self.attention(
            Q=I_embeddings.unsqueeze(1), K=IA_embeddings, V=IA_embeddings
        ) # self attention
        I_attention_scores = I_attention_scores.squeeze(1) # (batch_size, n_aspects)
        I_embeddings_aggregated = I_embeddings_aggregated.squeeze(1) # (batch_size, d_model)
        I_embeddings = I_embeddings_aggregated # no skip connection

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
            R_hat = self.overall_rating(torch.cat([U_embeddings, I_embeddings], dim=-1)).squeeze(1)
            #R_hat = torch.clamp(R_hat, min=self.config.min_rating, max=self.config.max_rating)
            losses = self.ratings_loss(R, R_hat, A_ratings, A_ratings_hat)
            _out.update({
                "overall_rating": R_hat,
                "aspects_ratings": A_ratings_hat,
                "losses": losses
            })

        if self.config.review_flag and (inference_flag or self.training_phase == 1):
            U_embeddings_words = self.user_words_proj(U_embeddings).unsqueeze(1) # (batch_size, 1, d_words)
            I_embeddings_words = self.item_words_proj(I_embeddings).unsqueeze(1) # (batch_size, 1, d_words)
            UA_embeddings_words = self.user_aspect_words_proj(UA_embeddings) # (batch_size, n_aspects, d_words)
            IA_embeddings_words = self.item_aspect_words_proj(IA_embeddings) # (batch_size, n_aspects, d_words)
            P_embeddings = torch.cat(
                [U_embeddings_words, I_embeddings_words, UA_embeddings_words, IA_embeddings_words], 
                dim=1
            ) # (batch_size, 2 * (1 + n_aspects), d_words)

            if not inference_flag:
                review_loss = self.text_module.decode(P_embeddings, review_tokens)
                losses = {"review": review_loss, "total": review_loss}
                _out.update({"losses": losses})

            else:
                review_tokens_hat = self.text_module.generate(P_embeddings)
                review = self.tokenizer.batch_decode(review_tokens_hat, skip_special_tokens=True)
                _out.update({'review': review})

        return _out


class AURACrossAggregation(nn.Module):
    """ AURA with cross-attention aggregation
    """

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
            self.text_module = text_module
            self.tokenizer = tokenizer

            self.user_words_proj = nn.Linear(config.d_model, config.d_words)
            self.item_words_proj = nn.Linear(config.d_model, config.d_words)
            self.user_aspect_words_proj = nn.Linear(config.d_model, config.d_words)
            self.item_aspect_words_proj = nn.Linear(config.d_model, config.d_words)
            #self.prompt_embedding = nn.Sequential(
            #    nn.Linear(2 * (1 + self.config.n_aspects) * config.d_words, config.n_prompt * config.d_words),
            #    nn.ReLU(),
            #    nn.Linear(config.n_prompt * config.d_words, config.n_prompt * config.d_words)
            #)

        self.training_phase = 0 # 0: ratings, 1: prompt, 2: text module + prompt
        self._init_weights()

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
            self.prompt_grad(True)
            self.ratings_grad(False)
            self.text_grad(False)
        elif phase == 2:
            self.text_grad(True)

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
        for param in self.user_words_proj.parameters():
            param.requires_grad = flag
        for param in self.item_words_proj.parameters():
            param.requires_grad = flag
        for param in self.user_aspect_words_proj.parameters():
            param.requires_grad = flag
        for param in self.item_aspect_words_proj.parameters():
            param.requires_grad = flag
        #for param in self.prompt_embedding.parameters():
        #    param.requires_grad = flag

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
            Q=U_embeddings.unsqueeze(1), K=IA_embeddings, V=UA_embeddings
        ) # cross attention
        U_attention_scores = U_attention_scores.squeeze(1) # (batch_size, n_aspects)
        U_embeddings_aggregated = U_embeddings_aggregated.squeeze(1) # (batch_size, d_model)
        U_embeddings = U_embeddings_aggregated # no skip connection

        I_embeddings_aggregated, I_attention_scores = self.item_attention(
            Q=I_embeddings.unsqueeze(1), K=UA_embeddings, V=IA_embeddings
        ) # cross attention
        I_attention_scores = I_attention_scores.squeeze(1) # (batch_size, n_aspects)
        I_embeddings_aggregated = I_embeddings_aggregated.squeeze(1) # (batch_size, d_model)
        I_embeddings = I_embeddings_aggregated # no skip connection

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
            R_hat = self.overall_rating(torch.cat([U_embeddings, I_embeddings], dim=-1)).squeeze(1)
            #R_hat = torch.clamp(R_hat, min=self.config.min_rating, max=self.config.max_rating)
            losses = self.ratings_loss(R, R_hat, A_ratings, A_ratings_hat)
            _out.update({
                "overall_rating": R_hat,
                "aspects_ratings": A_ratings_hat,
                "losses": losses
            })

        if self.config.review_flag and (inference_flag or self.training_phase == 1):
            U_embeddings_words = self.user_words_proj(U_embeddings).unsqueeze(1) # (batch_size, 1, d_words)
            I_embeddings_words = self.item_words_proj(I_embeddings).unsqueeze(1) # (batch_size, 1, d_words)
            UA_embeddings_words = self.user_aspect_words_proj(UA_embeddings) # (batch_size, n_aspects, d_words)
            IA_embeddings_words = self.item_aspect_words_proj(IA_embeddings) # (batch_size, n_aspects, d_words)
            P_embeddings = torch.cat(
                [U_embeddings_words, I_embeddings_words, UA_embeddings_words, IA_embeddings_words], 
                dim=1
            ) # (batch_size, 2 * (1 + n_aspects), d_words)

            if not inference_flag:
                review_loss = self.text_module.decode(P_embeddings, review_tokens)
                losses = {"review": review_loss, "total": review_loss}
                _out.update({"losses": losses})

            else:
                review_tokens_hat = self.text_module.generate(P_embeddings)
                review = self.tokenizer.batch_decode(review_tokens_hat, skip_special_tokens=True)
                _out.update({'review': review})

        return _out


class AURAAllShared(nn.Module):

    def __init__(self, config, text_module=None, tokenizer=None):
        super().__init__()
        self.config = config

        # Rating
        self.ratings_loss = RatingsLoss(config)

        self.user_embedding = nn.Embedding(config.n_users, config.d_model)
        self.item_embedding = nn.Embedding(config.n_items, config.d_model)
        self.attention = Attention(config.d_model)
        self.attention = Attention(config.d_model)

        self.aspects_embedding = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.ReLU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(config.d_model, config.d_model)
            ) for _ in range(config.n_aspects)
        ])

        self.aspects_rating = nn.Sequential(
            nn.Linear(2 * config.d_model, config.d_model),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(config.d_model, 1)
        )
        self.overall_rating = nn.Sequential(
            nn.Linear(2 * config.d_model, config.d_model),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(config.d_model, 1)
        )

        # Review
        if config.review_flag:
            self.text_module = text_module
            self.tokenizer = tokenizer

            self.user_words_proj = nn.Linear(config.d_model, config.d_words)
            self.item_words_proj = nn.Linear(config.d_model, config.d_words)
            self.user_aspect_words_proj = nn.Linear(config.d_model, config.d_words)
            self.item_aspect_words_proj = nn.Linear(config.d_model, config.d_words)
            #self.prompt_embedding = nn.Sequential(
            #    nn.Linear(2 * (1 + self.config.n_aspects) * config.d_words, config.n_prompt * config.d_words),
            #    nn.ReLU(),
            #    nn.Linear(config.n_prompt * config.d_words, config.n_prompt * config.d_words)
            #)

        self.training_phase = 0 # 0: ratings, 1: prompt, 2: text module + prompt
        self._init_weights()

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
            self.prompt_grad(True)
            self.ratings_grad(False)
            self.text_grad(False)
        elif phase == 2:
            self.text_grad(True)

    def ratings_grad(self, flag: bool=True):
        for param in self.user_embedding.parameters():
            param.requires_grad = flag
        for param in self.item_embedding.parameters():
            param.requires_grad = flag
        for param in self.attention.parameters():
            param.requires_grad = flag
        for param in self.aspects_embedding.parameters():
            param.requires_grad = flag
        for param in self.aspects_rating.parameters():
            param.requires_grad = flag
        for param in self.overall_rating.parameters():
            param.requires_grad = flag

    def prompt_grad(self, flag: bool=True):
        if not self.config.review_flag:
            return
        for param in self.user_words_proj.parameters():
            param.requires_grad = flag
        for param in self.item_words_proj.parameters():
            param.requires_grad = flag
        for param in self.user_aspect_words_proj.parameters():
            param.requires_grad = flag
        for param in self.item_aspect_words_proj.parameters():
            param.requires_grad = flag
        #for param in self.prompt_embedding.parameters():
        #    param.requires_grad = flag

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
            au_embedding = self.aspects_embedding[i](U_embeddings) # (batch_size, d_model)
            ai_embedding = self.aspects_embedding[i](I_embeddings) # (batch_size, d_model)
            UA_embeddings.append(au_embedding)
            IA_embeddings.append(ai_embedding)

            if inference_flag or self.training_phase == 0:
                a_rating = self.aspects_rating(torch.cat([au_embedding, ai_embedding], dim=-1)) # (batch_size,)
                #a_rating = torch.clamp(a_rating, min=self.config.min_rating, max=self.config.max_rating)
                A_ratings_hat.append(a_rating)

        UA_embeddings = torch.stack(UA_embeddings, dim=1) # (batch_size, n_aspects, d_model)
        IA_embeddings = torch.stack(IA_embeddings, dim=1) # (batch_size, n_aspects, d_model)
            
        U_embeddings_aggregated, U_attention_scores = self.attention(
            Q=U_embeddings.unsqueeze(1), K=UA_embeddings, V=UA_embeddings
        ) # self attention
        U_attention_scores = U_attention_scores.squeeze(1) # (batch_size, n_aspects)
        U_embeddings_aggregated = U_embeddings_aggregated.squeeze(1) # (batch_size, d_model)
        U_embeddings = U_embeddings_aggregated # no skip connection

        I_embeddings_aggregated, I_attention_scores = self.attention(
            Q=I_embeddings.unsqueeze(1), K=IA_embeddings, V=IA_embeddings
        ) # self attention
        I_attention_scores = I_attention_scores.squeeze(1) # (batch_size, n_aspects)
        I_embeddings_aggregated = I_embeddings_aggregated.squeeze(1) # (batch_size, d_model)
        I_embeddings = I_embeddings_aggregated # no skip connection

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
            R_hat = self.overall_rating(torch.cat([U_embeddings, I_embeddings], dim=-1)).squeeze(1)
            #R_hat = torch.clamp(R_hat, min=self.config.min_rating, max=self.config.max_rating)
            losses = self.ratings_loss(R, R_hat, A_ratings, A_ratings_hat)
            _out.update({
                "overall_rating": R_hat,
                "aspects_ratings": A_ratings_hat,
                "losses": losses
            })

        if self.config.review_flag and (inference_flag or self.training_phase == 1):
            U_embeddings_words = self.user_words_proj(U_embeddings).unsqueeze(1) # (batch_size, 1, d_words)
            I_embeddings_words = self.item_words_proj(I_embeddings).unsqueeze(1) # (batch_size, 1, d_words)
            UA_embeddings_words = self.user_aspect_words_proj(UA_embeddings) # (batch_size, n_aspects, d_words)
            IA_embeddings_words = self.item_aspect_words_proj(IA_embeddings) # (batch_size, n_aspects, d_words)
            P_embeddings = torch.cat(
                [U_embeddings_words, I_embeddings_words, UA_embeddings_words, IA_embeddings_words], 
                dim=1
            ) # (batch_size, 2 * (1 + n_aspects), d_words)

            if not inference_flag:
                review_loss = self.text_module.decode(P_embeddings, review_tokens)
                losses = {"review": review_loss, "total": review_loss}
                _out.update({"losses": losses})

            else:
                review_tokens_hat = self.text_module.generate(P_embeddings)
                review = self.tokenizer.batch_decode(review_tokens_hat, skip_special_tokens=True)
                _out.update({'review': review})

        return _out
