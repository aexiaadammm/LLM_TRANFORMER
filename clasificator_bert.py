# -*- coding: utf-8 -*-
"""
clasificator_bert.py — clasificatorul de paragrafe, de data asta pornind de la un model pre-antrenat (MiniLM, de tip BERT).

Compară, pe aceleași eseuri și cu aceeași evaluare ca train_clasificator.py (leave-one-essay-out):
  A. MiniLM „înghețat”: greutățile lui nu se schimbă; antrenăm doar stratul final (cap)
  B. fine-tuning: antrenăm stratul final ȘI ajustăm puțin greutățile lui MiniLM
și le pune lângă metoda pe cuvinte-cheie și Transformer-ul tău de la zero.

Rulare:  python clasificator_bert.py            (A + B; B durează, pe laptop ~15-30 de minute)
         python clasificator_bert.py --fara-ft  (doar A, câteva minute)
"""

import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from prelucrare import normalizeaza, pregateste, cuvinte_cheie, caracteristici_pozitie
from train_clasificator import citeste_date, raport, ETICHETE

MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
OUT_FILE = 'clasificator_bert.pt'
MAX_TOKENI = 256        # paragrafele mai lungi se taie (un token ≈ o bucată de cuvânt)

# setări
epoci_cap = 60          # A: doar stratul final, se antrenează repede
epoci_ft = 4            # B: fine-tuning, puține epoci ca să nu „uite” ce știa
batch_size = 8
lr_cap = 1e-3           # learning rate pentru stratul final
lr_encoder = 2e-5       # learning rate MIC pentru greutățile lui MiniLM
torch.manual_seed(1337)
random.seed(1337)


class Cap(nn.Module):
    """stratul nou, de la final: vectorul paragrafului + poziția lui -> etichetă"""
    def __init__(self, dim, num_classes, n_pos=3):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(dim + n_pos, num_classes)

    def forward(self, v, pos, targets=None):
        logits = self.linear(self.dropout(torch.cat([v, pos], dim=-1)))
        loss = F.cross_entropy(logits, targets) if targets is not None else None
        return logits, loss


class ClasificatorBert(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL)          # MiniLM, deja antrenat
        self.cap = Cap(self.encoder.config.hidden_size, num_classes)   # hidden_size = 384

    def vector(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state   # (B, T, 384)
        # TODO 1: mean pooling, ca în TransformerClassifier: media vectorilor tokenilor reali (fără padding)
        #   attention_mask are forma (B, T): 1 = token real, 0 = padding
        #   1. m = attention_mask.unsqueeze(-1).float()        -> forma (B, T, 1), ca să se poată înmulți cu h
        #   2. aduni vectorii tokenilor reali: (h * m).sum(1)
        #   3. împarți la câți tokeni reali are fiecare paragraf: m.sum(1).clamp(min=1)
        #   4. întorci rezultatul, cu forma (B, 384)
        pass

    def forward(self, input_ids, attention_mask, pos, targets=None):
        return self.cap(self.vector(input_ids, attention_mask), pos, targets)


def tokenizeaza(tokenizer, date):
    enc = tokenizer([d['text_bert'] for d in date], padding=True, truncation=True,
                    max_length=MAX_TOKENI, return_tensors='pt')
    pos = torch.tensor([caracteristici_pozitie(d['nr'], d['total']) for d in date])
    return enc['input_ids'], enc['attention_mask'], pos


def pas_antrenare(optimizer, loss):
    # TODO 2: pasul de antrenare, ca în train.py
    #   trei rânduri: golești gradienții vechi, calculezi gradienții noi din loss, actualizezi greutățile
    pass


# ---------------- A: MiniLM înghețat ----------------
@torch.no_grad()
def calculeaza_vectori(model, tokenizer, date):
    """vectorul fiecărui paragraf, o singură dată (MiniLM nu se schimbă în varianta A)"""
    model.eval()
    for i in range(0, len(date), batch_size):
        b = date[i:i + batch_size]
        idx, mask, _ = tokenizeaza(tokenizer, b)
        for d, v in zip(b, model.vector(idx, mask)):
            d['vector'] = v


def antreneaza_cap(train):
    cap = Cap(train[0]['vector'].shape[0], len(ETICHETE))      # doar stratul final, fără MiniLM
    # TODO 3: optimizatorul primește DOAR parametrii stratului final
    #   optimizer = torch.optim.AdamW( ... parametrii lui cap ..., lr=lr_cap)
    optimizer = None
    for ep in range(epoci_cap):
        random.shuffle(train)
        cap.train()
        for i in range(0, len(train), batch_size):
            b = train[i:i + batch_size]
            v = torch.stack([d['vector'] for d in b])
            pos = torch.tensor([caracteristici_pozitie(d['nr'], d['total']) for d in b])
            _, loss = cap(v, pos, torch.tensor([d['y'] for d in b]))
            pas_antrenare(optimizer, loss)
    return cap


@torch.no_grad()
def prezice_cap(cap, date):
    cap.eval()
    v = torch.stack([d['vector'] for d in date])
    pos = torch.tensor([caracteristici_pozitie(d['nr'], d['total']) for d in date])
    logits, _ = cap(v, pos)
    return logits.argmax(-1).tolist()


# ---------------- B: fine-tuning ----------------
def antreneaza_ft(train, tokenizer, verbose=False):
    model = ClasificatorBert(len(ETICHETE))
    # dicționarul de cuvinte al lui MiniLM (250.000 de tokeni x 384) rămâne înghețat:
    # e cea mai mare parte din model și n-avem destule date ca să-l schimbăm
    model.encoder.embeddings.word_embeddings.weight.requires_grad = False
    # TODO 4: două grupuri de parametri, cu learning rate diferit:
    #   restul lui MiniLM -> lr_encoder (mic), stratul final -> lr_cap
    #   AdamW primește o listă de dicționare {'params': ..., 'lr': ...}:
    #     1. parametrii lui model.encoder care au requires_grad == True (o listă cu if), cu lr_encoder
    #     2. parametrii lui model.cap, cu lr_cap
    optimizer = None
    for ep in range(epoci_ft):
        random.shuffle(train)
        model.train()
        total = 0
        for i in range(0, len(train), batch_size):
            b = train[i:i + batch_size]
            idx, mask, pos = tokenizeaza(tokenizer, b)
            _, loss = model(idx, mask, pos, torch.tensor([d['y'] for d in b]))
            pas_antrenare(optimizer, loss)
            total += loss.item() * len(b)
        if verbose:
            print(f'  epoca {ep}: loss {total / len(train):.4f}')
    return model


@torch.no_grad()
def prezice_ft(model, tokenizer, date):
    model.eval()
    pred = []
    for i in range(0, len(date), batch_size):
        idx, mask, pos = tokenizeaza(tokenizer, date[i:i + batch_size])
        logits, _ = model(idx, mask, pos)
        pred += logits.argmax(-1).tolist()
    return pred


if __name__ == '__main__':
    fine_tuning = '--fara-ft' not in sys.argv
    date = citeste_date()
    import csv
    with open('date/paragrafe_etichetate.csv', encoding='utf-8-sig') as f:   # textul original, cu diacritice
        originale = [r for r in csv.DictReader(f, delimiter=';') if r['eticheta'] in ETICHETE]
    for d, r in zip(date, originale):
        d['text_bert'] = normalizeaza(r['text'])               # MiniLM vrea textul normal, cu majuscule și diacritice

    eseuri = sorted(set(d['eseu'] for d in date), key=int)
    de_testat = [e for e in eseuri if any(d['eseu'] == e and d['sursa'] != 'generat_AI' for d in date)]
    print(f'{len(date)} paragrafe; evaluare pe {len(de_testat)} eseuri scrise de oameni')

    print('Încarc MiniLM...')
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    baza = ClasificatorBert(len(ETICHETE))
    calculeaza_vectori(baza, tokenizer, date)

    y_true, y_kw, y_a, y_b = [], [], [], []
    for e in de_testat:
        train = [d for d in date if d['eseu'] != e]
        test = [d for d in date if d['eseu'] == e]
        y_true += [d['y'] for d in test]
        y_kw += [ETICHETE.index(cuvinte_cheie(d['text'], d['nr'], d['total'])) for d in test]
        y_a += prezice_cap(antreneaza_cap(list(train)), test)
        msg = f'  eseul {e}: A {sum(a == b for a, b in zip(y_true[-len(test):], y_a[-len(test):]))}/{len(test)}'
        if fine_tuning:
            y_b += prezice_ft(antreneaza_ft(list(train), tokenizer), tokenizer, test)
            msg += f', B {sum(a == b for a, b in zip(y_true[-len(test):], y_b[-len(test):]))}/{len(test)}'
        print(msg, flush=True)

    rez = {'cuvinte-cheie': raport('Cuvinte-cheie', y_true, y_kw),
           'A (MiniLM înghețat)': raport('A: MiniLM înghețat + strat final', y_true, y_a)}
    if fine_tuning:
        rez['B (fine-tuning)'] = raport('B: fine-tuning MiniLM', y_true, y_b)
    try:
        rez['Transformer de la zero'] = torch.load('clasificator.pt', map_location='cpu')['acc_transformer']
    except Exception:
        pass

    print('\n=== Comparație (paragrafe recunoscute corect) ===')
    for k, v in rez.items():
        if v is not None:
            print(f'  {k:26s} {100 * v:.0f}%')

    # TODO 5: varianta mai bună dintre A și B se antrenează pe toate datele și se salvează
    #   best = 'B' dacă s-a rulat fine-tuning-ul ȘI rez['B (fine-tuning)'] > rez['A (MiniLM înghețat)'], altfel 'A'
    best = 'A'
    print(f'\n=== Model final: varianta {best}, pe toate eseurile ===')
    if best == 'A':
        stare = antreneaza_cap(list(date)).state_dict()        # doar stratul final, câțiva KB
    else:
        stare = antreneaza_ft(list(date), tokenizer, verbose=True).state_dict()   # tot modelul, ~470 MB
    torch.save({'varianta': best, 'model_baza': MODEL, 'stare': stare, 'etichete': ETICHETE,
                'acuratete': {k: v for k, v in rez.items() if v is not None}, 'nr_eseuri': len(de_testat)},
               OUT_FILE)
    print('salvat în', OUT_FILE)
