import torch
import torch.nn as nn

class NewTokenEmbeddings(nn.Module):
    def __init__(
            self, 
            old_embeddings : nn.Embedding = None,
            embed_dim : int = 1024,
            old_vocab_size : int = 248077,
            tokenizer : any = None,
            mean_subwords: bool = False,
            subwords_id_list : list = None,
            new_tokens : list = None
        ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.old_embeddings = old_embeddings # [248077, 1024]
        self.old_vocab_size = old_vocab_size
        self.old_embeddings.requires_grad_(False)

        self.num_new_tokens = len(new_tokens)
        self.new_embeddings = nn.Embedding(self.num_new_tokens, embed_dim) # [237, 1024]
        print(f"old vocab size - {self.old_vocab_size}")
        print(f"new token size - {self.num_new_tokens}")

        with torch.no_grad():
            if mean_subwords:
                if not subwords_id_list:
                    raise ValueError(f"not subwords id list given ...")
                for i, token in enumerate(new_tokens):
                    token_id = tokenizer.convert_tokens_to_ids(token)
                    avg = self.old_embeddings.weight[subwords_id_list[i]].mean(dim=0)
                    self.new_embeddings.weight[token_id - self.old_vocab_size] = avg
            else:
                avg = self.old_embeddings.weight.mean(dim=0)
                for i, token in enumerate(new_tokens):
                    token_id = tokenizer.convert_tokens_to_ids(token)
                    self.new_embeddings.weight[token_id - self.old_vocab_size] = avg

    def forward(self, input_ids):
        old_ids = torch.clamp(input_ids, min=0, max=self.old_vocab_size - 1)
        new_ids = torch.clamp(input_ids - self.old_vocab_size, min=0, max=self.num_new_tokens - 1)

        old_vectors = self.old_embeddings(old_ids)
        new_vectors = self.new_embeddings(new_ids)

        is_old = (input_ids < self.old_vocab_size).unsqueeze(-1).to(old_vectors.dtype) 

        return old_vectors * is_old + new_vectors * (1.0 - is_old)
    
class NewTokenOutput(nn.Module):
    def __init__(
            self, 
            old_lm_head : nn.Linear = None,
            embed_dim : int = 1024,
            old_vocab_size : int = 248077,
            tokenizer : any = None,
            mean_subwords: bool = False,
            subwords_id_list : list = None,
            new_tokens : list = None
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.old_lm_head = old_lm_head
        self.old_vocab_size = old_vocab_size
        self.old_lm_head.requires_grad_(False)

        self.num_new_tokens = len(new_tokens)
        self.new_lm_head = nn.Linear(embed_dim, self.num_new_tokens, bias = False)

        with torch.no_grad():
            if mean_subwords:
                if not subwords_id_list:
                    raise ValueError(f"not subwords id list given ...")
                for i, token in enumerate(new_tokens):
                    token_id = tokenizer.convert_tokens_to_ids(token)
                    avg = self.old_lm_head.weight[subwords_id_list[i]].mean(dim=0)
                    self.new_lm_head.weight[token_id - self.old_vocab_size] = avg
            else:
                avg = self.old_lm_head.weight.mean(dim=0)
                for i, token in enumerate(new_tokens):
                    token_id = tokenizer.convert_tokens_to_ids(token)
                    self.new_lm_head.weight[token_id - self.old_vocab_size] = avg
            
    @property
    def weight(self):
        return self.new_lm_head.weight

    def forward(self, hidden_states):
        # hidden_states - [batch_size, seq_len, 1024] 
        target_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(self.old_lm_head.weight.dtype)
        old_token_logits = self.old_lm_head(hidden_states).to(target_dtype)  # [batch_size, seq_len, 248077]
        old_token_logits = old_token_logits[..., :self.old_vocab_size]
        new_token_logits = self.new_lm_head(hidden_states).to(target_dtype)  # [batch_size, seq_len, 233]
        logits = torch.cat([old_token_logits, new_token_logits], dim=-1) # [batch_size, seq_len, 248310]
        return logits