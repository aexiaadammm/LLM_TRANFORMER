# -*- coding: utf-8 -*-
"""verifica un text argumentativ după baremul Subiectului I B"""

import re
from prelucrare import imparte_in_paragrafe, pregateste, cuvinte_cheie_arg

CONECTORI = ['in primul rand', 'in al doilea rand', 'de asemenea', 'in plus', 'mai mult',
             'deoarece', 'pentru ca', 'prin urmare', 'asadar', 'in concluzie', 'de exemplu',
             'totusi', 'pe de o parte', 'pe de alta parte', 'un prim exemplu', 'un al doilea exemplu']


def verifica_argumentativ(text, fragment=''):
    paragrafe = imparte_in_paragrafe(text or '')
    if not paragrafe:
        return [], 'Lipește textul argumentativ.'

    total = len(paragrafe)
    etichete = [cuvinte_cheie_arg(p, i + 1, total) for i, p in enumerate(paragrafe)]
    tabel = [[i + 1, e, p[:100]] for i, (e, p) in enumerate(zip(etichete, paragrafe))]

    lipsa, ok = [], []

    # nr de cuvinte (exemplu completat)
    cuvinte = sum(len(p.split()) for p in paragrafe)
    if cuvinte >= 150:
        ok.append(f'Lungimea: {cuvinte} de cuvinte (minimum 150).')
    else:
        lipsa.append(f'Textul are doar {cuvinte} de cuvinte; baremul cere minimum 150.')

    # TODO 7: opinia
    if 'opinie' in etichete:
        ok.append('Opinia: există.')
    else:
        lipsa.append('Nu am găsit opinia. Formulează clar ce crezi („Consider că…”).')

    # TODO 8: argumentele

    nr_arg = etichete.count('argument')
    if nr_arg >= 2:
        ok.append(f'Argumente: {nr_arg} (baremul cere două).')
    else:
        lipsa.append(f'Am găsit doar {nr_arg} argument(e); baremul cere două, fiecare dezvoltat.')

    # TODO 9: concluzia

    if 'concluzie' in etichete:
        ok.append('Concluzia: există.')
    else:
        lipsa.append('Lipsește concluzia.')

    #TODO 10 conectorii
    tot_textul = pregateste(' '.join(paragrafe))
    gasiti = [c for c in CONECTORI if c in tot_textul]
    if len(gasiti) >= 3:
        ok.append(f'Conectori: {len(gasiti)} ({", ".join(gasiti)}).')
    else:
        lipsa.append(f'Folosești puțini conectori ({len(gasiti)}). Leagă argumentele cu „în primul rând”, '
                     '„de asemenea”, „prin urmare”.')


    # TODO 11: raportarea la textul suport // pune cu copy paste textul
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
            ok.append(f'Raportarea la text: {arg_legate} argument(e) par să folosească fragmentul dat.')
        else:
            lipsa.append('Argumentele nu par să se raporteze la fragmentul dat. Folosește idei sau '
                         'exemple din text în dezvoltarea argumentelor (3 puncte în barem).')
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