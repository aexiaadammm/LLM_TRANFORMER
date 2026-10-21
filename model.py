# -*- coding: utf-8 -*-
"""
model.py — arhitectura GPT-ului (aceeași ca în main.py), fără date și fără antrenare.
Setările (n_embed, n_head, ...) se dau ca parametri, ca să poată fi salvate
împreună cu modelul și refolosite în app.py.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """un singur cap de self-attention

    causal=True  -> decoder (GPT): fiecare literă vede doar trecutul (masca tril)
    causal=False -> encoder (clasificator): fiecare literă vede tot textul
    """

    def __init__(self, n_embed, head_size, block_size, dropout, causal=True):
        super().__init__()
        self.causal = causal
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))  # masca
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pad_mask=None):
        B, T, C = x.shape
        k = self.key(x)      # (B, T, head_size)  "ce sunt eu"
        q = self.query(x)    # (B, T, head_size)  "ce caut"
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5           # (B, T, T) notele / √d_k
        if self.causal:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # blocăm viitorul
        if pad_mask is not None:
            # pad_mask: (B, T), True = literă reală, False = umplutură (padding) -> nu ne uităm la ea
            wei = wei.masked_fill(~pad_mask[:, None, :], float('-inf'))
        wei = F.softmax(wei, dim=-1)                                  # note -> procente
        wei = self.dropout(wei)
        v = self.value(x)    # (B, T, head_size)  "ce dau mai departe"
        return wei @ v       # (B, T, head_size)


class MultiHeadAttention(nn.Module):
    """mai multe capete în paralel; rezultatele se lipesc (concat) și se proiectează"""

    def __init__(self, n_embed, num_heads, head_size, block_size, dropout, causal=True):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embed, head_size, block_size, dropout, causal)
                                    for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pad_mask=None):
        out = torch.cat([h(x, pad_mask) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """fiecare literă își prelucrează singură informația adunată"""

    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """un bloc Transformer: comunicare (atenție), apoi calcul (feed-forward)"""

    def __init__(self, n_embed, n_head, block_size, dropout, causal=True):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_embed, n_head, head_size, block_size, dropout, causal)
        self.ffwd = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x, pad_mask=None):
        x = x + self.sa(self.ln1(x), pad_mask)   # conexiune reziduală
        x = x + self.ffwd(self.ln2(x))   # conexiune reziduală
        return x


class GPTLanguageModel(nn.Module):

    def __init__(self, vocab_size, n_embed=32, n_head=4, n_layer=3, block_size=8, dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head, block_size, dropout)
                                      for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)                                  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, C)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                                   # (B, T, vocab)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """continuă textul din idx cu max_new_tokens caractere noi"""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]          # doar ultimele block_size caractere
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature       # temperature: mic = sigur, mare = creativ
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')  # păstrăm doar cele mai probabile top_k
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class TransformerClassifier(nn.Module):
    """Același Transformer, folosit ca ENCODER pentru clasificare.

    Diferențe față de GPTLanguageModel:
      1. fără mască cauzală: fiecare literă vede tot paragraful (și înainte, și după)
      2. paragrafele au lungimi diferite, deci cele scurte sunt completate cu PAD;
         masca de padding face ca atenția să ignore aceste umpluturi
      3. la final nu prezicem litera următoare, ci facem media vectorilor tuturor literelor
         (un singur vector pentru tot paragraful) și alegem eticheta din barem
      4. primim și poziția paragrafului în eseu (primul? ultimul? la ce procent din eseu?),
         pentru că introducerea e de obicei la început, iar concluzia la final
    """

    def __init__(self, vocab_size, num_classes, n_embed=64, n_head=4, n_layer=2,
                 block_size=300, dropout=0.1, n_pos_features=3):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.ModuleList([Block(n_embed, n_head, block_size, dropout, causal=False)
                                     for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(n_embed + n_pos_features, num_classes)   # vectorul paragrafului -> etichete

    def forward(self, idx, pad_mask, pos_features, targets=None):
        # idx: (B, T) literele, pad_mask: (B, T) True = literă reală, pos_features: (B, 3)
        B, T = idx.shape
        x = self.token_embedding_table(idx) + self.position_embedding_table(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x, pad_mask)
        x = self.ln_f(x)                                          # (B, T, C)
        m = pad_mask.unsqueeze(-1).float()                        # (B, T, 1)
        vec = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)      # media doar pe literele reale: (B, C)
        vec = torch.cat([self.dropout(vec), pos_features], dim=-1)
        logits = self.head(vec)                                   # (B, num_classes)
        loss = None if targets is None else F.cross_entropy(logits, targets)
        return logits, loss
