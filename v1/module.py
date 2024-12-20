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
        self.review_rating_loss = nn.MSELoss()

    def forward(self, R: torch.Tensor, R_hat: torch.Tensor,
                A_ratings: torch.Tensor, A_ratings_hat: torch.Tensor,
                reviews_ratings: torch.Tensor, reviews_ratings_hat: torch.Tensor) -> Dict[str, torch.Tensor]:

        overall_rating_loss = self.overall_rating_loss(R_hat, R)
        aspect_rating_loss = self.aspect_rating_loss(A_ratings_hat.flatten(), A_ratings.flatten())
        review_rating_loss = self.review_rating_loss(reviews_ratings_hat.flatten(), reviews_ratings.flatten())

        total_loss = (
            self.config.alpha * overall_rating_loss +
            self.config.beta * aspect_rating_loss +
            self.config.gamma * review_rating_loss
        )

        return {
            "total": total_loss,
            "overall_rating": overall_rating_loss,
            "aspect_rating": aspect_rating_loss,
            "review_rating": review_rating_loss
        }


class SentenceEmbedding(nn.Module):

    def __init__(self, config, encoder=None):
        super().__init__()
        self.config = config
        self.encoder = encoder
        #self.sentence_attention = Attention(config.d_words)

    def encode(self, sentence_tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode(sentence_tokens, mask)

    def forward(self, E_embeddings: torch.Tensor,
                sentence_tokens: torch.Tensor, mask: torch.Tensor=None) -> Dict[str, torch.Tensor]:
        # E_embeddings: (batch_size, d_words)
        # sentence_tokens, mask: (batch_size, seq_len)

        tokens_embeddings = self.encode(sentence_tokens, mask) # (batch_size, seq_len, d_words)
        sentence_embeddings, attention_scores = attention_function(
            Q=E_embeddings.unsqueeze(1), K=tokens_embeddings, V=tokens_embeddings, mask=mask
        )
        attention_scores = attention_scores.squeeze(1) # (batch_size, seq_len)
        sentence_embeddings = sentence_embeddings.squeeze(1) # (batch_size, d_words)

        _out = {
            "embeddings": sentence_embeddings,
            "attention": attention_scores
        }
        return _out


class DocumentEmbedding(nn.Module):

    def __init__(self, config, encoder=None):
        super().__init__()
        self.config = config
        #self.document_attention = Attention(config.d_words)
        self.sentence_embedding = SentenceEmbedding(config, encoder)
        self.overall_rating = nn.Sequential(
            nn.Linear(self.config.d_words, self.config.d_model),
            nn.ReLU(),
            nn.Linear(self.config.d_model, 1)
        )

    def forward(self, E_embeddings: torch.Tensor,
                document_tokens: torch.Tensor, mask: torch.Tensor=None) -> Dict[str, torch.Tensor]:
        # E_embeddings: (batch_size, d_words)
        # document_tokens, attention_mask: (batch_size, n_sentences, seq_len)

        batch_size, n_sentences, seq_len = document_tokens.size()

        document_tokens_flat = document_tokens.view(-1, seq_len) # (batch_size * n_sentences, seq_len)
        mask_flat = mask.view(-1, seq_len) if mask is not None else None
        E_embeddings_flat = E_embeddings.repeat(n_sentences, 1) # (batch_size * n_sentences, d_words)

        sentence_outputs = self.sentence_embedding(E_embeddings_flat, document_tokens_flat, mask_flat)
        sentences_embeddings = sentence_outputs["embeddings"].view(batch_size, n_sentences, -1) # (batch_size, n_sentences, d_words)

        document_mask = mask.sum(dim=-1) # (batch_size, n_sentences)
        document_embeddings, attention_scores = attention_function(
            Q=E_embeddings.unsqueeze(1) , K=sentences_embeddings, V=sentences_embeddings, mask=document_mask
        )
        attention_scores = attention_scores.squeeze(1) # (batch_size, n_sentences)
        document_embeddings = document_embeddings.squeeze(1) # (batch_size, d_words)

        rating = self.overall_rating(document_embeddings).squeeze(1) # (batch_size,)
        rating = torch.clamp(rating, min=self.config.min_rating, max=self.config.max_rating)

        _out = {
            "embeddings": document_embeddings,
            "attention": attention_scores,
            "rating": rating,
            "sentence" : {
                "embeddings": sentences_embeddings,
                "attention": sentence_outputs["attention"].view(batch_size, n_sentences, seq_len)
            }
        }
        return _out


class AURA(nn.Module):

    def __init__(self, config, text_module, tokenizer):
        super().__init__()
        self.config = config
        self.text_module = text_module
        self.tokenizer = tokenizer
        self.ratings_loss = RatingsLoss(config)

        self.user_embedding = nn.Embedding(config.n_users, config.d_model)
        self.item_embedding = nn.Embedding(config.n_items, config.d_model)

        self.user_document_embedding = DocumentEmbedding(config, text_module)
        self.item_document_embedding = DocumentEmbedding(config, text_module)

        self.user_words_proj = nn.Linear(config.d_model, config.d_words)
        self.item_words_proj = nn.Linear(config.d_model, config.d_words)

        self.user_aspects_embedding = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.ReLU(),
                nn.Linear(config.d_model, config.d_model)
            ) for _ in range(config.n_aspects)
        ])
        self.item_aspects_embedding = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.ReLU(),
                nn.Linear(config.d_model, config.d_model)
            ) for _ in range(config.n_aspects)
        ])

        self.aspects_rating = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * config.d_model, config.d_model),
                nn.ReLU(),
                nn.Linear(config.d_model, 1)
            ) for _ in range(config.n_aspects)
        ])

        self.overall_rating = nn.Sequential(
            nn.Linear(2 * config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, 1)
        )

        self.prompt_embedding = nn.Sequential(
            nn.Linear(2 * config.d_words, config.n_prompt * config.d_words),
            nn.ReLU(),
            nn.Linear(config.n_prompt * config.d_words, config.n_prompt * config.d_words)
        )

        self.training_phase = 0 # 0: ratings, 1: prompt, 2: text module

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
        #elif phase == 2:
        #    self.text_grad()

    def ratings_grad(self, flag: bool=True):
        for param in self.user_embedding.parameters():
            param.requires_grad = flag
        for param in self.item_embedding.parameters():
            param.requires_grad = flag
        for param in self.user_document_embedding.parameters():
            param.requires_grad = flag
        for param in self.item_document_embedding.parameters():
            param.requires_grad = flag
        for param in self.user_words_proj.parameters():
            param.requires_grad = flag
        for param in self.item_words_proj.parameters():
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
        for param in self.prompt_embedding.parameters():
            param.requires_grad = flag

    def text_grad(self, flag: bool=True):
        for param in self.text_module.parameters():
            param.requires_grad = flag

    def forward(self, U_ids: torch.Tensor, I_ids: torch.Tensor,
                U_document_tokens: torch.Tensor=None, U_mask: torch.Tensor=None, U_ratings: torch.Tensor=None,
                I_document_tokens: torch.Tensor=None, I_mask: torch.Tensor=None, I_ratings: torch.Tensor=None,
                R: torch.Tensor=None, A_ratings: torch.Tensor=None,
                UI_review_tokens: torch.Tensor=None,
                inference_flag: bool=False) -> Dict[str, torch.Tensor]:
        # U_ids, I_ids: (batch_size,)
        # U_document_tokens, I_documents, U_mask, I_mask: (batch_size, n_sentences, seq_len)
        # UI_review_tokens: (batch_size, seq_len)

        batch_size = U_ids.size(0)

        U_embeddings = self.user_embedding(U_ids) # (batch_size, d_model)
        I_embeddings = self.item_embedding(I_ids) # (batch_size, d_model)

        U_embeddings_words = self.user_words_proj(U_embeddings) # (batch_size, d_words)
        I_embeddings_words = self.item_words_proj(I_embeddings) # (batch_size, d_words)

        U_outputs = self.user_document_embedding(U_embeddings_words, U_document_tokens, U_mask)
        I_outputs = self.item_document_embedding(I_embeddings_words, I_document_tokens, I_mask)

        _out = {}

        if inference_flag or self.training_phase == 0:
            UA_embeddings = []
            IA_embeddings = []
            A_ratings_hat = []
            for i in range(self.config.n_aspects):
                au_embedding = self.user_aspects_embedding[i](U_embeddings) # (batch_size, d_model)
                ai_embedding = self.item_aspects_embedding[i](I_embeddings) # (batch_size, d_model)
                a_rating = self.aspects_rating[i](torch.cat([au_embedding, ai_embedding], dim=-1)) # (batch_size,)
                a_rating = torch.clamp(a_rating, min=self.config.min_rating, max=self.config.max_rating)

                UA_embeddings.append(au_embedding)
                IA_embeddings.append(ai_embedding)
                A_ratings_hat.append(a_rating)

            UA_embeddings = torch.stack(UA_embeddings, dim=1) # (batch_size, n_aspects, d_model)
            IA_embeddings = torch.stack(IA_embeddings, dim=1) # (batch_size, n_aspects, d_model)
            A_ratings_hat = torch.stack(A_ratings_hat, dim=1).squeeze(2) # (batch_size, n_aspects)

            U_embeddings_aggregated, U_attention_scores = attention_function(
                Q=U_embeddings.unsqueeze(1), K=IA_embeddings, V=UA_embeddings
            )
            U_attention_scores = U_attention_scores.squeeze(1) # (batch_size, n_aspects)
            U_embeddings_aggregated = U_embeddings_aggregated.squeeze(1) # (batch_size, d_model)
            U_embeddings = U_embeddings + U_embeddings_aggregated

            I_embeddings_aggregated, I_attention_scores = attention_function(
                Q=I_embeddings.unsqueeze(1), K=UA_embeddings, V=IA_embeddings
            )
            I_attention_scores = I_attention_scores.squeeze(1) # (batch_size, n_aspects)
            I_embeddings_aggregated = I_embeddings_aggregated.squeeze(1) # (batch_size, d_model)
            I_embeddings = I_embeddings + I_embeddings_aggregated

            R_hat = self.overall_rating(torch.cat([U_embeddings, I_embeddings], dim=-1)).squeeze(1)
            R_hat = torch.clamp(R_hat, min=self.config.min_rating, max=self.config.max_rating)

            _out.update({
                "overall_rating": R_hat,
                "aspects_ratings": A_ratings_hat,
                "user": {
                    "attention": U_attention_scores,
                    "document": U_outputs
                },
                "item": {
                    "attention": I_attention_scores,
                    "document": I_outputs
                }
            })

            U_ratings_hat = U_outputs["rating"]
            I_ratings_hat = I_outputs["rating"]
            reviews_ratings_hat = torch.cat([U_ratings_hat, I_ratings_hat], dim=-1)
            reviews_ratings = torch.cat([U_ratings, I_ratings], dim=-1)

            losses = self.ratings_loss(R, R_hat, A_ratings, A_ratings_hat, reviews_ratings, reviews_ratings_hat)
            _out.update({"losses": losses})

        if inference_flag or self.training_phase == 1:
            P_embeddings = torch.cat([U_embeddings_words, I_embeddings_words], dim=1)
            P_embeddings = self.prompt_embedding(P_embeddings).view(batch_size, self.config.n_prompt, self.config.d_words)

            if not inference_flag:
                review_loss = self.text_module.decode(P_embeddings, UI_review_tokens)
                losses = {"review": review_loss, "total": review_loss}
                _out.update({"losses": losses})

            else:
                UI_review_tokens_hat = self.text_module.generate(P_embeddings)
                review = self.tokenizer.batch_decode(UI_review_tokens_hat, skip_special_tokens=True)
                _out.update({'review': review})

        return _out
