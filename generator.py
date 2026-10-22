# -*- coding: utf-8 -*-
"""
generator.py — încarcă GPT-ul antrenat de train.py (model.pt) și generează text.
Folosit de server.py. Dacă model.pt nu există, disponibil = False.
"""

import os

import torch

from model import GPTLanguageModel

MODEL_FILE = 'model.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

disponibil = False
if os.path.exists(MODEL_FILE):
    _ck = torch.load(MODEL_FILE, map_location=device)
    _chars = _ck['chars']
    _model = GPTLanguageModel(**_ck['config']).to(device)
    _model.load_state_dict(_ck['model'])
    _model.eval()                                    # modul de folosire: fără dropout
    _stoi = {ch: i for i, ch in enumerate(_chars)}
    _itos = {i: ch for i, ch in enumerate(_chars)}
    disponibil = True


def genereaza(text_start, lungime=300, temperatura=0.8, top_k=0):
    """continuă text_start; întoarce (text generat, caractere ignorate)"""
    if not disponibil:
        return None, []
    text_start = (text_start or '').replace('ş', 'ș').replace('Ş', 'Ș').replace('ţ', 'ț').replace('Ţ', 'Ț')
    necunoscute = sorted(set(c for c in text_start if c not in _stoi))
    cunoscute = [c for c in text_start if c in _stoi]
    ids = [_stoi[c] for c in cunoscute] if cunoscute else ([_stoi['\n']] if '\n' in _stoi else [0])
    context = torch.tensor([ids], dtype=torch.long, device=device)
    lungime = max(10, min(int(lungime), 2000))
    temperatura = max(0.1, min(float(temperatura), 2.0))
    out = _model.generate(context, max_new_tokens=lungime, temperature=temperatura,
                          top_k=int(top_k) if top_k else None)
    text = ''.join(_itos[i] for i in out[0].tolist())
    return (text if cunoscute else text.lstrip('\n')), necunoscute
