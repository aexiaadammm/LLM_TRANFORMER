# -*- coding: utf-8 -*-
"""verifica un text argumentativ după baremul Subiectului I B"""

import re
from prelucrare import imparte_in_paragrafe, pregateste, cuvinte_cheie_arg

CONECTORI = ['in primul rand', 'in al doilea rand', 'de asemenea', 'in plus', 'mai mult',
             'deoarece', 'pentru ca', 'prin urmare', 'asadar', 'in concluzie', 'de exemplu',
             'totusi', 'pe de o parte', 'pe de alta parte', 'un prim exemplu', 'un al doilea exemplu']

NUME_ARG = {
    'introducere': 'Introducere',
    'opinie': 'Opinie',
    'argument': 'Argument',
    'concluzie': 'Concluzie',
    'altele': 'Altceva',
}


def analizeaza_argumentativ(text, fragment=''):
    """analiza textului argumentativ ca date (folosită de Gradio și de server.py)"""
    paragrafe = imparte_in_paragrafe(text or '')
    if not paragrafe:
        return None

    total = len(paragrafe)
    etichete = [cuvinte_cheie_arg(p, i + 1, total) for i, p in enumerate(paragrafe)]
    cerinte = []   # fiecare: nume, stare ('ok' / 'atentie' / 'lipsa'), mesaj

    def cerinta(nume, stare, mesaj):
        cerinte.append({'nume': nume, 'stare': stare, 'mesaj': mesaj})

    # nr de cuvinte
    cuvinte = sum(len(p.split()) for p in paragrafe)
    if cuvinte >= 150:
        cerinta('Lungime', 'ok', f'{cuvinte} de cuvinte (minimum 150).')
    else:
        cerinta('Lungime', 'lipsa', f'Textul are doar {cuvinte} de cuvinte; baremul cere minimum 150.')

    # opinia
    if 'opinie' in etichete:
        cerinta('Opinie', 'ok', 'Există.')
    else:
        cerinta('Opinie', 'lipsa', 'Nu am găsit opinia. Formulează clar ce crezi („Consider că…”).')

    # argumentele
    nr_arg = etichete.count('argument')
    if nr_arg >= 2:
        cerinta('Argumente', 'ok', f'{nr_arg} (baremul cere două).')
    else:
        cerinta('Argumente', 'lipsa', f'Am găsit doar {nr_arg} argument(e); baremul cere două, fiecare dezvoltat.')

    # concluzia
    if 'concluzie' in etichete:
        cerinta('Concluzie', 'ok', 'Există.')
    else:
        cerinta('Concluzie', 'lipsa', 'Lipsește concluzia.')

    # conectorii
    tot_textul = pregateste(' '.join(paragrafe))
    gasiti = [c for c in CONECTORI if c in tot_textul]
    if len(gasiti) >= 3:
        cerinta('Conectori', 'ok', f'{len(gasiti)} ({", ".join(gasiti)}).')
    else:
        cerinta('Conectori', 'lipsa', f'Folosești puțini conectori ({len(gasiti)}). Leagă argumentele cu '
                '„în primul rând”, „de asemenea”, „prin urmare”.')

    # raportarea la textul suport (doar dacă elevul a lipit fragmentul)
    if fragment:
        frag = pregateste(fragment)
        cuv_fragment = set(re.findall(r'\w{5,}', frag))     # cuvintele de minimum 5 litere din fragment
        semne = r'\b(in text|textul dat|textul suport|fragment|autorul|conform textului|potrivit textului)'

        arg_legate = 0
        for p, e in zip(paragrafe, etichete):
            if e != 'argument':
                continue
            t = pregateste(p)
            comune = cuv_fragment & set(re.findall(r'\w{5,}', t))   # cuvinte comune cu fragmentul
            if len(comune) >= 3 or re.search(semne, t):
                arg_legate += 1

        if arg_legate >= 1:
            cerinta('Raportarea la text', 'ok', f'{arg_legate} argument(e) par să folosească fragmentul dat.')
        else:
            cerinta('Raportarea la text', 'lipsa', 'Argumentele nu par să se raporteze la fragmentul dat. '
                    'Folosește idei sau exemple din text în dezvoltarea argumentelor (3 puncte în barem).')

    return {
        'paragrafe': [{'nr': i + 1, 'eticheta': e, 'nume': NUME_ARG.get(e, e), 'text': p, 'cuvinte': len(p.split())}
                      for i, (e, p) in enumerate(zip(etichete, paragrafe))],
        'cerinte': cerinte,
        'cuvinte': cuvinte,
        'indeplinite': sum(c['stare'] == 'ok' for c in cerinte),
        'total': len(cerinte),
        'metoda': 'Analiză pe reguli: cuvinte-cheie, conectori și cuvinte comune cu textul suport.',
    }


def verifica_argumentativ(text, fragment=''):
    """pentru interfața Gradio: întoarce (tabel, raport text)"""
    a = analizeaza_argumentativ(text, fragment)
    if a is None:
        return [], 'Lipește textul argumentativ.'
    tabel = [[p['nr'], p['eticheta'], p['text'][:100]] for p in a['paragrafe']]
    lipsa = [f'{c["nume"]}: {c["mesaj"]}' for c in a['cerinte'] if c['stare'] != 'ok']
    ok = [f'{c["nume"]}: {c["mesaj"]}' for c in a['cerinte'] if c['stare'] == 'ok']
    raport = ''
    if lipsa:
        raport += '### Ce lipsește\n' + '\n'.join('- ' + x for x in lipsa) + '\n\n'
    if ok:
        raport += '### În regulă\n' + '\n'.join('- ' + x for x in ok)
    return tabel, raport


if __name__ == '__main__':
    t, r = verifica_argumentativ(open('test_arg.txt', encoding='utf-8').read())
    for rand in t:
        print(rand)
    print(r)
