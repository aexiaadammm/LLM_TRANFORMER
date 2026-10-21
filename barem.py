# -*- coding: utf-8 -*-
"""
barem.py — verifică un eseu după baremul Subiectului al III-lea (eseul despre particularitățile unui text).

1. împarte eseul în paragrafe
2. pune fiecărui paragraf o etichetă (introducere, incadrare, tema, secventa, element_structura, concluzie)
   - cu Transformer-ul tău (clasificator.pt), dacă la evaluare a fost mai bun decât metoda simplă
   - altfel cu metoda simplă, pe cuvinte-cheie
3. numără ce a găsit și spune ce lipsește din barem
"""

import os

import torch

from model import TransformerClassifier
from prelucrare import imparte_in_paragrafe, pregateste, cuvinte_cheie, batch_tensori

CLASIFICATOR = 'clasificator.pt'

NUME = {
    'introducere': 'Introducere',
    'incadrare': 'Încadrare (trăsătură)',
    'tema': 'Tema',
    'secventa': 'Secvență / scenă',
    'element_structura': 'Element de structură / limbaj',
    'concluzie': 'Concluzie',
    'altele': 'Altceva',
}

# ---------------- încărcarea clasificatorului (dacă există) ----------------
_model, _stoi, _etichete, _info = None, None, None, {}
if os.path.exists(CLASIFICATOR):
    ck = torch.load(CLASIFICATOR, map_location='cpu')
    _info = {k: ck.get(k) for k in ('acc_transformer', 'acc_cuvinte_cheie', 'nr_paragrafe', 'nr_eseuri')}
    acc_tr, acc_kw = _info['acc_transformer'], _info['acc_cuvinte_cheie']
    if acc_tr is not None and acc_kw is not None and acc_tr > acc_kw:   # îl folosim doar dacă e mai bun
        _model = TransformerClassifier(**ck['config'])
        _model.load_state_dict(ck['model'])
        _model.eval()
        _stoi, _etichete = ck['stoi'], ck['etichete']


def metoda():
    """ce metodă folosește aplicația și cât de bine a mers la evaluare"""
    tr, kw = _info.get('acc_transformer'), _info.get('acc_cuvinte_cheie')
    scor = ''
    if tr is not None and kw is not None:
        scor = (f' La evaluare, pe {_info["nr_eseuri"]} eseuri ({_info["nr_paragrafe"]} paragrafe): '
                f'Transformer {100 * tr:.0f}%, cuvinte-cheie {100 * kw:.0f}% paragrafe recunoscute corect.')
    if _model is not None:
        return 'Clasificator: Transformer-ul antrenat de tine.' + scor
    return 'Clasificator: metoda simplă, pe cuvinte-cheie (mai bună deocamdată).' + scor


@torch.no_grad()
def eticheteaza(paragrafe):
    total = len(paragrafe)
    if _model is None:
        return [cuvinte_cheie(p, i + 1, total) for i, p in enumerate(paragrafe)]
    idx, mask, pos = batch_tensori([(pregateste(p), i + 1, total) for i, p in enumerate(paragrafe)], _stoi)
    logits, _ = _model(idx, mask, pos)
    return [_etichete[i] for i in logits.argmax(-1).tolist()]


def verifica(eseu):
    """întoarce (tabel cu eticheta fiecărui paragraf, raport cu ce e în regulă și ce lipsește)"""
    paragrafe = imparte_in_paragrafe(eseu or '')
    if not paragrafe:
        return [], 'Lipește un eseu, cu paragrafele despărțite prin rânduri noi.'

    etichete = eticheteaza(paragrafe)
    tabel = [[i + 1, NUME[e], p[:120] + ('…' if len(p) > 120 else '')]
             for i, (e, p) in enumerate(zip(etichete, paragrafe))]
    n = {e: etichete.count(e) for e in NUME}
    cuvinte = sum(len(p.split()) for p in paragrafe)

    ok, lipsa, atentie = [], [], []

    # redactare: lungime și părți componente
    if cuvinte >= 400:
        ok.append(f'Lungimea: {cuvinte} de cuvinte (minimum 400).')
    else:
        lipsa.append(f'Eseul are doar {cuvinte} de cuvinte; baremul cere minimum 400.')
    for parte in ('introducere', 'concluzie'):
        (ok if n[parte] else lipsa).append(
            f'{NUME[parte]}: ' + ('există.' if n[parte] else 'lipsește (contează la redactare, „părțile componente”).'))
    if n['concluzie'] and etichete[-1] != 'concluzie':
        atentie.append('Concluzia nu pare să fie ultimul paragraf.')

    # cerința 1: două trăsături de încadrare (6 puncte)
    if n['incadrare'] >= 2:
        ok.append(f'Încadrarea: {n["incadrare"]} paragrafe cu trăsături (baremul cere două trăsături).')
    elif n['incadrare'] == 1:
        atentie.append('Încadrarea: un singur paragraf cu trăsături. Baremul cere **două trăsături**; '
                       'verifică dacă le ai pe amândouă, ilustrate cu exemple din text.')
    else:
        lipsa.append('Încadrarea: n-am găsit nicio trăsătură care să încadreze opera într-un curent, '
                     'o perioadă sau o specie (cerința 1, 6 puncte).')

    # cerința 2: tema + două secvențe (6 puncte)
    (ok if n['tema'] else lipsa).append(
        'Tema: ' + ('există.' if n['tema'] else 'n-am găsit un paragraf despre temă (cerința 2).'))
    if n['secventa'] >= 2:
        ok.append(f'Secvențe: {n["secventa"]} paragrafe. Verifică să fie **două** secvențe diferite, '
                  'nu una singură continuată pe mai multe paragrafe.')
    elif n['secventa'] == 1:
        atentie.append('Secvențe: un singur paragraf. Baremul cere comentarea a **două** secvențe sau scene.')
    else:
        lipsa.append('Secvențe: n-am găsit nicio secvență comentată. Baremul cere **două** secvențe sau scene '
                     'relevante pentru temă (cerința 2, 6 puncte).')

    # cerința 3: două elemente de structură / limbaj (6 puncte)
    if n['element_structura'] >= 2:
        ok.append(f'Elemente de structură: {n["element_structura"]} paragrafe (baremul cere două).')
    elif n['element_structura'] == 1:
        atentie.append('Elemente de structură: unul singur. Baremul cere **două** (de ex. titlul, incipitul, '
                       'finalul, conflictul, perspectiva narativă, motivele, figurile de stil).')
    else:
        lipsa.append('Elemente de structură: n-am găsit niciunul. Baremul cere **două** componente de structură '
                     'și/sau de limbaj (cerința 3, 6 puncte).')

    raport = []
    if lipsa:
        raport.append('### Ce lipsește\n' + '\n'.join('- ' + x for x in lipsa))
    if atentie:
        raport.append('### De verificat\n' + '\n'.join('- ' + x for x in atentie))
    if ok:
        raport.append('### În regulă\n' + '\n'.join('- ' + x for x in ok))
    raport.append('_' + metoda() + ' Etichetele pot greși: folosește-le ca ajutor, nu ca notă._')
    return tabel, '\n\n'.join(raport)


if __name__ == '__main__':
    import sys
    text = open(sys.argv[1], encoding='utf-8').read()
    tabel, raport = verifica(text)
    for r in tabel:
        print(r[0], r[1].ljust(32), r[2][:70])
    print()
    print(raport)
