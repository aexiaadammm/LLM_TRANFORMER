# -*- coding: utf-8 -*-
"""
raportare_model.py — verifică dacă un argument folosește textul suport, cu un model pre-antrenat.

1. citește perechile (fragment, argument, legat) din date/
2. creează automat perechi negative (argument + alt fragment)
3. modelul pre-antrenat dă un scor de similaritate pentru fiecare pereche
4. alege pragul pe datele de antrenare (sursa != gasit)
5. compară modelul cu regula veche pe datele reale (sursa == gasit)
6. salvează pragul în prag_raportare.json
"""

import csv
import json
import re

from sentence_transformers import SentenceTransformer, util

from prelucrare import pregateste

MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
FIS_PRAG = 'prag_raportare.json'


def citeste_csv(cale):
    with open(cale, encoding='utf-8-sig') as f:
        return list(csv.DictReader(f, delimiter=';'))


def propozitii(text):
    """împarte un text în propoziții (după . ! ?); păstrează doar pe cele de minimum 4 cuvinte"""
    bucati = re.split(r'(?<=[.!?])\s+', text)
    return [b for b in bucati if len(b.split()) >= 4] or [text]


def perechi():
    fragmente = {r['fragment_id']: r['text'] for r in citeste_csv('date/fragmente.csv')}
    randuri = citeste_csv('date/raportare.csv')
    P = []

    # TODO 1: pentru fiecare rând r din randuri, adauga in P un dicționar cu cheile:
    for r in randuri:
        P.append({'fragment': fragmente[r['fragment_id']], 'argument': r['argument'],
                  'legat': int(r['legat']), 'sursa': r['sursa'], 'tip': 'scris'})



    # TODO 2: negative automate: pune fiecare argument lângă ALT fragment -> legat = 0
    #legat=0 => drag nu foloseste textul suport deci e e x din viata pers
    for r in randuri:
        alte = [k for k in fragmente if k != r['fragment_id']
                and any(x['fragment_id'] == k and x['sursa'] == r['sursa'] for x in randuri)]
        if alte:
            P.append({'fragment': fragmente[alte[0]], 'argument': r['argument'],
                      'legat': 0, 'sursa': r['sursa'], 'tip': 'automat'})

    return P


def scor_model(model, fragment, argument):
    # TODO 3: scorul = cea mai mare similaritate dintre argument și o propoziție din fragment
    v_prop = model.encode(propozitii(fragment))
    v_arg = model.encode(propozitii(argument))
    return util.cos_sim(v_arg, v_prop).max().item()



def regula(fragment, argument):
    """regula veche din barem_arg.py: >= 3 cuvinte comune (de minimum 5 litere) sau o trimitere explicită"""
    semne = r'\b(in text|textul dat|textul suport|fragment|autorul|conform textului|potrivit textului)'
    comune = set(re.findall(r'\w{5,}', pregateste(fragment))) & set(re.findall(r'\w{5,}', pregateste(argument)))
    return int(len(comune) >= 3 or bool(re.search(semne, pregateste(argument))))


def acuratete(predictii, etichete):
    # TODO 4a: ce fracție din predicții sunt egale cu etichetele (un număr între 0 și 1)
    return sum(p == e for p, e in zip(predictii, etichete)) / len(etichete)


if __name__ == '__main__':
    print('Încarc modelul...')
    model = SentenceTransformer(MODEL)
    P = perechi()
    for p in P:
        p['scor'] = scor_model(model, p['fragment'], p['argument'])

    antrenare = [p for p in P if p['sursa'] != 'gasit']
    test = [p for p in P if p['sursa'] == 'gasit']
    print(f'{len(antrenare)} perechi de antrenare, {len(test)} perechi de test (scrise de oameni)')

    # TODO 4 incerc pragurile pe antrenare si pastrez predictia cea mai buna
    cel_mai_bun, acc_max = 0.5, -1
    for i in range(10, 91):
        prag = i / 100
        acc = acuratete([int(p['scor'] >= prag) for p in antrenare], [p['legat'] for p in antrenare])
        if acc > acc_max:
            cel_mai_bun, acc_max = prag, acc

    print(f'Pragul ales: {cel_mai_bun:.2f} (acuratețe pe antrenare {100 * acc_max:.0f}%)')

    # TODO 5: acuratețea modelului pe test, cu pragul ales
    #   la fel ca în TODO 4, dar pe lista test și doar cu pragul cel_mai_bun
    acc_model = acuratete([int(p['scor'] >= cel_mai_bun) for p in test], [p['legat'] for p in test])


    # TODO 6: acuratețea regulii vechi pe test
    #   predicția regulii pentru p: regula(p['fragment'], p['argument'])
    acc_regula = acuratete([regula(p['fragment'], p['argument']) for p in test], [p['legat'] for p in test])


    print(f'\nPe datele reale: model {100 * acc_model:.0f}%, regula veche {100 * acc_regula:.0f}%\n')
    for p in test:
        print(f'{p["scor"]:.2f}  corect={p["legat"]}  model={int(p["scor"] >= cel_mai_bun)}  '
              f'regula={regula(p["fragment"], p["argument"])}  {p["tip"]:7}  {p["argument"][:60]}...')

    # TODO 7: salvează pragul și rezultatele, ca să le folosească aplicația
    with open(FIS_PRAG, 'w', encoding='utf-8') as f:
        json.dump({'model': MODEL, 'prag': cel_mai_bun, 'acc_model': acc_model, 'acc_regula': acc_regula,
                   'nr_test': len(test)}, f, ensure_ascii=False, indent=2)

    print(f'\nAm salvat {FIS_PRAG}')
