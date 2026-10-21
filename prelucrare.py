# -*- coding: utf-8 -*-
"""
prelucrare.py — pregătirea textului pentru clasificator, folosită la fel la antrenare și în aplicație.
"""

import re
import random

import torch

PAD, UNK = 0, 1          # 0 = umplutură (padding), 1 = caracter necunoscut
MAX_LEN = 120            # câte caractere citim din fiecare paragraf (primele propoziții spun cel mai mult)

FARA_DIACRITICE = str.maketrans('ăâîșțşţĂÂÎȘȚŞŢ', 'aaistst' + 'AAISTST')


def normalizeaza(text):
    """ş/ţ cu sedilă -> ș/ț cu virgulă, spații uniforme"""
    text = text.replace('ş', 'ș').replace('Ş', 'Ș').replace('ţ', 'ț').replace('Ţ', 'Ț')
    text = text.replace('\xa0', ' ')
    return re.sub(r'\s+', ' ', text).strip()


def scoate_diacritice(text):
    return text.translate(FARA_DIACRITICE)


def pregateste(text):
    """forma pe care o vede clasificatorul: litere mici, fără diacritice
    (mulți elevi scriu fără diacritice; așa „Tema” și „tema”, „secvență” și „secventa” arată la fel)"""
    return scoate_diacritice(normalizeaza(text).lower())


def cuvinte_cheie(text, nr, total):
    """metoda simplă, de comparație: caută cuvinte-cheie la începutul paragrafului"""
    t = pregateste(text)[:150]
    inceput = t[:60]          # primele cuvinte spun cel mai clar despre ce e paragraful
    concl = r'\b(in concluzie|asadar|prin urmare|astfel,)'
    secv = r'\b(secvent|(?<!pe )scen|episod|strofa|fragment)'   # „pe scenă” = locul unde se joacă piesa, nu o scenă
    incad = (r'\b(trasatur|se incadreaza|apartine curentului|specific'
             r'|(element|aspect)\w* (romantic|realist|modernist|simbolist|traditionalis|clasic))')
    tema = r'\b(tema|temele|tematica)\b'
    elem = r'\b(titlu|incipit|final|conflict|perspectiv|narator|personaj|motiv|timp|spati|limbaj|prozodi|compozi)'

    if re.search(concl, t):
        return 'concluzie'
    # 1. ce anunță începutul paragrafului
    for eticheta, regula in (('secventa', secv), ('incadrare', incad), ('tema', tema), ('element_structura', elem)):
        if re.search(regula, inceput):
            return eticheta
    # 2. ultimul paragraf e de obicei concluzia
    if nr == total:
        return 'concluzie'
    # 3. ce apare mai departe în paragraf
    for eticheta, regula in (('tema', tema), ('secventa', secv), ('element_structura', elem), ('incadrare', incad)):
        if re.search(regula, t):
            return eticheta
    if nr <= 2:
        return 'introducere'
    return 'element_structura'


def imparte_in_paragrafe(eseu):
    """un paragraf = un bloc de text despărțit de restul printr-un rând nou"""
    parti = re.split(r'\n\s*\n|\n', eseu.strip())
    return [normalizeaza(p) for p in parti if len(p.split()) >= 3]   # sărim rândurile foarte scurte


def construieste_vocabular(texte):
    """toate caracterele din texte (și variantele fără diacritice); 0 și 1 sunt rezervate"""
    chars = set()
    for t in texte:
        chars.update(t)
        chars.update(scoate_diacritice(t))
    chars = sorted(chars)
    return {ch: i + 2 for i, ch in enumerate(chars)}


def codifica(text, stoi, max_len=MAX_LEN):
    ids = [stoi.get(c, UNK) for c in text[:max_len]]
    return ids if ids else [UNK]


def caracteristici_pozitie(nr, total):
    """poziția paragrafului în eseu: e primul? e ultimul? cât de departe e (0 = început, 1 = final)?"""
    rel = (nr - 1) / max(total - 1, 1)
    return [1.0 if nr == 1 else 0.0, 1.0 if nr == total else 0.0, rel]


def batch_tensori(exemple, stoi, max_len=MAX_LEN, augmentare=False):
    """exemple: listă de (text, nr, total). Întoarce idx, pad_mask, pos_features."""
    seqs, pos = [], []
    for text, nr, total in exemple:
        if augmentare and random.random() < 0.5:
            text = scoate_diacritice(text)       # mulți elevi scriu fără diacritice
        seqs.append(codifica(text, stoi, max_len))
        pos.append(caracteristici_pozitie(nr, total))
    T = max(len(s) for s in seqs)
    idx = torch.full((len(seqs), T), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        idx[i, :len(s)] = torch.tensor(s)
    return idx, idx != PAD, torch.tensor(pos, dtype=torch.float)


def cuvinte_cheie_arg(text, nr, total):
    """etichetează un paragraf dintr-un text argumentativ"""
    t = pregateste(text)  # litere mici, fără diacritice

    if nr <= 2 and re.search(
            r'\b(consider ca|dupa parerea mea|in opinia mea|sunt de parere|cred ca|din punctul meu de vedere)', t):
        return 'opinie'
    if nr == 1:
        return 'introducere'
    if nr == total or re.search(r'\b(in concluzie|asadar|prin urmare)', t):
        return 'concluzie'


    # argumentele
    if re.search(r'\b(in primul rand|in al doilea rand|pe de o parte|un argument|de asemenea|pe de alta parte|un prim argument|un al doilea argument)', t):
        return 'argument'
    # orice altceva
    return 'argument'
