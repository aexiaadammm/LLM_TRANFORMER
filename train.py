# -*- coding: utf-8 -*-
"""
train.py — antrenează GPT-ul și îl salvează în model.pt (împreună cu vocabularul și setările).
Rulează-l o singură dată; după aceea folosește app.py ca să generezi text.
"""

import os
import urllib.request

import torch

from model import GPTLanguageModel

# ---------------- setări ----------------
DATA_FILE = 'input.txt'   # textul de antrenare (pune aici alt fișier .txt pentru alt text, de ex. românesc)
DATA_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
OUT_FILE = 'model.pt'

# setări mici, pentru laptop (câteva minute pe procesor)
batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 500
eval_iters = 200
learning_rate = 1e-3
n_embed = 32
n_head = 4
n_layer = 3
dropout = 0.0

# setările modelului mare din video (doar pe GPU, de ex. Colab T4):
# batch_size = 64; block_size = 256; max_iters = 5000; learning_rate = 3e-4
# n_embed = 384; n_head = 6; n_layer = 6; dropout = 0.2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)
print('rulez pe:', device)

# ---------------- date ----------------
if not os.path.exists(DATA_FILE):
    print('descarc', DATA_URL)
    urllib.request.urlretrieve(DATA_URL, DATA_FILE)

with open(DATA_FILE, 'r', encoding='utf-8') as f:
    text = f.read()

# unificăm diacriticele românești (ş/ţ cu sedilă -> ș/ț cu virgulă); la textul englez nu schimbă nimic
text = text.replace('ş', 'ș').replace('Ş', 'Ș').replace('ţ', 'ț').replace('Ţ', 'Ț')

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
print(f'text: {len(text)} caractere, vocabular: {vocab_size} caractere')

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# ---------------- model ----------------
config = dict(vocab_size=vocab_size, n_embed=n_embed, n_head=n_head,
              n_layer=n_layer, block_size=block_size, dropout=dropout)
model = GPTLanguageModel(**config).to(device)
print(sum(p.numel() for p in model.parameters()), 'parametri')

# antrenenz
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for step in range(max_iters):
    if step % eval_interval == 0 or step == max_iters - 1:
        losses = estimate_loss()
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# salvare
torch.save({'model': model.state_dict(), 'config': config, 'chars': chars}, OUT_FILE)
print('model salvat în', OUT_FILE)

model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=300, temperature=0.8)[0].tolist()))
