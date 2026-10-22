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


def analizeaza(eseu):
    """analiza eseului ca date: eticheta fiecărui paragraf și starea fiecărei cerințe din barem.
    Folosită și de interfața Gradio (verifica), și de server.py (frontend-ul web)."""
    paragrafe = imparte_in_paragrafe(eseu or '')
    if not paragrafe:
        return None

    etichete = eticheteaza(paragrafe)
    n = {e: etichete.count(e) for e in NUME}
    cuvinte = sum(len(p.split()) for p in paragrafe)
    cerinte = []   # fiecare: nume, stare ('ok' / 'atentie' / 'lipsa'), mesaj

    def cerinta(nume, stare, mesaj):
        cerinte.append({'nume': nume, 'stare': stare, 'mesaj': mesaj})

    # redactare: lungime și părți componente
    if cuvinte >= 400:
        cerinta('Lungime', 'ok', f'{cuvinte} de cuvinte (minimum 400).')
    else:
        cerinta('Lungime', 'lipsa', f'Eseul are doar {cuvinte} de cuvinte; baremul cere minimum 400.')
    cerinta('Introducere', 'ok' if n['introducere'] else 'lipsa',
            'Există.' if n['introducere'] else 'Lipsește (contează la redactare, „părțile componente”).')

    # cerința 1: două trăsături de încadrare (6 puncte)
    if n['incadrare'] >= 2:
        cerinta('Încadrare', 'ok', f'{n["incadrare"]} paragrafe cu trăsături (baremul cere două trăsături).')
    elif n['incadrare'] == 1:
        cerinta('Încadrare', 'atentie', 'Un singur paragraf cu trăsături. Baremul cere **două trăsături**; '
                'verifică dacă le ai pe amândouă, ilustrate cu exemple din text.')
    else:
        cerinta('Încadrare', 'lipsa', 'N-am găsit nicio trăsătură care să încadreze opera într-un curent, '
                'o perioadă sau o specie (cerința 1, 6 puncte).')

    # cerința 2: tema + două secvențe (6 puncte)
    cerinta('Tema', 'ok' if n['tema'] else 'lipsa',
            'Există.' if n['tema'] else 'N-am găsit un paragraf despre temă (cerința 2).')
    if n['secventa'] >= 2:
        cerinta('Secvențe', 'ok', f'{n["secventa"]} paragrafe. Verifică să fie **două** secvențe diferite, '
                'nu una singură continuată pe mai multe paragrafe.')
    elif n['secventa'] == 1:
        cerinta('Secvențe', 'atentie', 'Un singur paragraf. Baremul cere comentarea a **două** secvențe sau scene.')
    else:
        cerinta('Secvențe', 'lipsa', 'N-am găsit nicio secvență comentată. Baremul cere **două** secvențe sau '
                'scene relevante pentru temă (cerința 2, 6 puncte).')

    # cerința 3: două elemente de structură / limbaj (6 puncte)
    if n['element_structura'] >= 2:
        cerinta('Elemente de structură', 'ok', f'{n["element_structura"]} paragrafe (baremul cere două).')
    elif n['element_structura'] == 1:
        cerinta('Elemente de structură', 'atentie', 'Unul singur. Baremul cere **două** (de ex. titlul, '
                'incipitul, finalul, conflictul, perspectiva narativă, motivele, figurile de stil).')
    else:
        cerinta('Elemente de structură', 'lipsa', 'N-am găsit niciunul. Baremul cere **două** componente de '
                'structură și/sau de limbaj (cerința 3, 6 puncte).')

    # concluzia
    if not n['concluzie']:
        cerinta('Concluzie', 'lipsa', 'Lipsește (contează la redactare, „părțile componente”).')
    elif etichete[-1] != 'concluzie':
        cerinta('Concluzie', 'atentie', 'Există, dar nu pare să fie ultimul paragraf.')
    else:
        cerinta('Concluzie', 'ok', 'Există.')

    return {
        'paragrafe': [{'nr': i + 1, 'eticheta': e, 'nume': NUME[e], 'text': p, 'cuvinte': len(p.split())}
                      for i, (e, p) in enumerate(zip(etichete, paragrafe))],
        'cerinte': cerinte,
        'cuvinte': cuvinte,
        'indeplinite': sum(c['stare'] == 'ok' for c in cerinte),
        'total': len(cerinte),
        'metoda': metoda(),
    }


def raport_markdown(cerinte, nota=''):
    """transformă lista de cerințe în raportul text afișat în Gradio"""
    raport = []
    for stare, titlu in (('lipsa', 'Ce lipsește'), ('atentie', 'De verificat'), ('ok', 'În regulă')):
        rand = [f'{c["nume"]}: {c["mesaj"]}' for c in cerinte if c['stare'] == stare]
        if rand:
            raport.append(f'### {titlu}\n' + '\n'.join('- ' + x for x in rand))
    if nota:
        raport.append(nota)
    return '\n\n'.join(raport)


def verifica(eseu):
    """pentru interfața Gradio: întoarce (tabel cu eticheta fiecărui paragraf, raport text)"""
    a = analizeaza(eseu)
    if a is None:
        return [], 'Lipește un eseu, cu paragrafele despărțite prin rânduri noi.'
    tabel = [[p['nr'], p['nume'], p['text'][:120] + ('…' if len(p['text']) > 120 else '')] for p in a['paragrafe']]
    nota = '_' + a['metoda'] + ' Etichetele pot greși: folosește-le ca ajutor, nu ca notă._'
    return tabel, raport_markdown(a['cerinte'], nota)


if __name__ == '__main__':
    import sys
    text = open(sys.argv[1], encoding='utf-8').read()
    tabel, raport = verifica(text)
    for r in tabel:
        print(r[0], r[1].ljust(32), r[2][:70])
    print()
    print(raport)
