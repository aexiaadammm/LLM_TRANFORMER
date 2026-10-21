# -*- coding: utf-8 -*-
"""
train_clasificator.py — antrenează clasificatorul de paragrafe (ce element din barem acoperă fiecare paragraf).

1. Citește date/paragrafe_etichetate.csv
2. EVALUARE: îl testează cinstit, pe eseuri pe care nu le-a văzut (leave-one-essay-out):
   pentru fiecare eseu, antrenează pe celelalte și verifică pe el. Îl compară cu o metodă simplă
   bazată pe cuvinte-cheie, ca să vedem dacă Transformer-ul chiar aduce ceva.
3. Antrenează modelul final pe TOATE eseurile și îl salvează în clasificator.pt

Rulare:  python train_clasificator.py            (evaluare + model final)
         python train_clasificator.py --rapid    (doar modelul final, fără evaluare)
"""

import csv
import random
import re
import sys
from collections import Counter, defaultdict

import torch

from model import TransformerClassifier
from prelucrare import (pregateste, cuvinte_cheie, construieste_vocabular, batch_tensori, MAX_LEN)

DATA_FILE = 'date/paragrafe_etichetate.csv'
OUT_FILE = 'clasificator.pt'
ETICHETE = ['introducere', 'incadrare', 'tema', 'secventa', 'element_structura', 'concluzie', 'altele']

# setări (mici, pentru laptop)
n_embed = 48
n_head = 4
n_layer = 2
dropout = 0.1
epoci = 40
batch_size = 16
learning_rate = 2e-3
torch.manual_seed(1337)
random.seed(1337)


# ---------------- date ----------------
def citeste_date():
    with open(DATA_FILE, encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f, delimiter=';'))
    total = Counter(r['eseu_id'] for r in rows)
    date = []
    for r in rows:
        if r['eticheta'] not in ETICHETE:
            print('etichetă necunoscută, sar rândul:', r['eticheta'], r['text'][:40])
            continue
        date.append(dict(eseu=r['eseu_id'], sursa=r.get('sursa') or 'gasit', text=pregateste(r['text']), nr=int(r['nr_paragraf']),
                         total=total[r['eseu_id']], y=ETICHETE.index(r['eticheta'])))
    return date


# metoda simplă de comparație (cuvinte_cheie) e în prelucrare.py, ca s-o folosească și aplicația


# ---------------- Transformer ----------------
def antreneaza(train, stoi, verbose=False):
    model = TransformerClassifier(len(stoi) + 2, len(ETICHETE), n_embed=n_embed, n_head=n_head,
                                  n_layer=n_layer, block_size=MAX_LEN, dropout=dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for ep in range(epoci):
        random.shuffle(train)
        model.train()
        total_loss = 0
        for i in range(0, len(train), batch_size):
            b = train[i:i + batch_size]
            idx, mask, pos = batch_tensori([(d['text'], d['nr'], d['total']) for d in b], stoi, augmentare=False)
            y = torch.tensor([d['y'] for d in b])
            _, loss = model(idx, mask, pos, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(b)
        if verbose and (ep % 10 == 0 or ep == epoci - 1):
            print(f'  epoca {ep}: loss {total_loss / len(train):.4f}')
    return model


@torch.no_grad()
def prezice(model, date, stoi):
    model.eval()
    idx, mask, pos = batch_tensori([(d['text'], d['nr'], d['total']) for d in date], stoi)
    logits, _ = model(idx, mask, pos)
    return logits.argmax(-1).tolist()


def raport(nume, y_true, y_pred):
    """afișează rezultatele și întoarce procentul de paragrafe corecte"""
    corecte = sum(a == b for a, b in zip(y_true, y_pred))
    print(f'\n{nume}: {corecte}/{len(y_true)} paragrafe corecte = {100 * corecte / len(y_true):.1f}%')
    pe_eticheta = defaultdict(lambda: [0, 0])
    for a, b in zip(y_true, y_pred):
        pe_eticheta[a][1] += 1
        pe_eticheta[a][0] += (a == b)
    for i, e in enumerate(ETICHETE):
        c, n = pe_eticheta[i]
        if n:
            print(f'   {e:18s} {c:3d}/{n:<3d} ({100 * c / n:.0f}%)')
    return corecte / len(y_true)


if __name__ == '__main__':
    date = citeste_date()
    eseuri = sorted(set(d['eseu'] for d in date), key=int)
    print(f'{len(date)} paragrafe din {len(eseuri)} eseuri')
    print('distribuția etichetelor:', dict(Counter(ETICHETE[d['y']] for d in date)))

    acc_tr = acc_kw = None
    if '--rapid' not in sys.argv:
        # testăm doar pe eseurile scrise de oameni; cele generate cu AI le folosim doar la antrenare,
        # ca să nu ne umfle rezultatul
        de_testat = [e for e in eseuri if any(d['eseu'] == e and d['sursa'] != 'generat_AI' for d in date)]
        print(f'\n=== Evaluare pe {len(de_testat)} eseuri scrise de oameni: '
              'pentru fiecare, antrenez pe celelalte și testez pe el ===')
        y_true, y_tr, y_kw = [], [], []
        for e in de_testat:
            train = [d for d in date if d['eseu'] != e]
            test = [d for d in date if d['eseu'] == e]
            stoi = construieste_vocabular([d['text'] for d in train])
            model = antreneaza(list(train), stoi)
            pred = prezice(model, test, stoi)
            kw = [ETICHETE.index(cuvinte_cheie(d['text'], d['nr'], d['total'])) for d in test]
            y_true += [d['y'] for d in test]
            y_tr += pred
            y_kw += kw
            ok = sum(a == b for a, b in zip([d['y'] for d in test], pred))
            print(f'  eseul {e}: {ok}/{len(test)} corecte')
        acc_kw = raport('Metoda simplă (cuvinte-cheie)', y_true, y_kw)
        acc_tr = raport('Transformer-ul tău', y_true, y_tr)
        if acc_tr > acc_kw:
            print('\n-> Transformer-ul e mai bun: aplicația îl va folosi pe el.')
        else:
            print('\n-> Deocamdată metoda simplă e mai bună (prea puține date pentru Transformer).'
                  '\n   Aplicația o va folosi pe ea; adaugă eseuri și reantrenează.')

    print('\n=== Modelul final, antrenat pe toate eseurile ===')
    stoi = construieste_vocabular([d['text'] for d in date])
    model = antreneaza(list(date), stoi, verbose=True)
    config = dict(vocab_size=len(stoi) + 2, num_classes=len(ETICHETE), n_embed=n_embed,
                  n_head=n_head, n_layer=n_layer, block_size=MAX_LEN, dropout=dropout)
    torch.save({'model': model.state_dict(), 'config': config, 'stoi': stoi, 'etichete': ETICHETE,
                'acc_transformer': acc_tr, 'acc_cuvinte_cheie': acc_kw,
                'nr_paragrafe': len([d for d in date if d['sursa'] != 'generat_AI']),
                'nr_eseuri': len([e for e in eseuri if any(d['eseu'] == e and d['sursa'] != 'generat_AI' for d in date)])},
               OUT_FILE)
    print('clasificator salvat în', OUT_FILE)
