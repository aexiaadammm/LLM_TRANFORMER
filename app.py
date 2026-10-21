# -*- coding: utf-8 -*-
"""
app.py — interfața web a proiectului, cu două taburi:
  1. „Verifică eseul”: lipești un eseu, iar aplicația spune ce element din barem acoperă fiecare paragraf
     și ce lipsește (folosește barem.py și, dacă există, clasificator.pt)
  2. „Generează text”: GPT-ul tău continuă un început de text (folosește model.pt, dacă există)

Pornire:  python app.py            (pagina se deschide la http://127.0.0.1:7860)
          python app.py --share    (în plus, un link public temporar)
"""

import os
import sys

import torch
import gradio as gr

from model import GPTLanguageModel
import barem
import barem_arg

MODEL_FILE = 'model.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------- generatorul (opțional: doar dacă ai rulat train.py) ----------------
model = None
if os.path.exists(MODEL_FILE):
    checkpoint = torch.load(MODEL_FILE, map_location=device)
    chars = checkpoint['chars']
    model = GPTLanguageModel(**checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()   # modul de folosire: fără dropout
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    decode = lambda l: ''.join([itos[i] for i in l])
    print(f'generator încărcat: {sum(p.numel() for p in model.parameters())} parametri')
else:
    print('nu găsesc model.pt: tabul de generare e dezactivat (rulează întâi train.py)')

print(barem.metoda())


def genereaza(text_start, lungime, temperatura, top_k):
    if model is None:
        return '', 'Nu există model.pt. Rulează întâi `python train.py`.'
    text_start = (text_start or '').replace('ş', 'ș').replace('Ş', 'Ș').replace('ţ', 'ț').replace('Ţ', 'Ț')
    necunoscute = sorted(set(c for c in text_start if c not in stoi))
    cunoscute = [c for c in text_start if c in stoi]
    ids = [stoi[c] for c in cunoscute] if cunoscute else ([stoi['\n']] if '\n' in stoi else [0])
    context = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(context, max_new_tokens=int(lungime),
                         temperature=float(temperatura), top_k=int(top_k) if top_k else None)
    text = decode(out[0].tolist())
    if not cunoscute:
        text = text.lstrip('\n')
    nota = ''
    if necunoscute:
        nota = 'Caractere ignorate (nu apar în textul de antrenare): ' + ' '.join(repr(c) for c in necunoscute)
    return text, nota


# ---------------- interfața ----------------
with gr.Blocks(title='BacHelper') as demo:
    gr.Markdown('## BacHelper\nVerifică-ți eseul după barem, în loc să tocești comentarii.')

    with gr.Tab('Verifică eseul'):
        gr.Markdown('Lipește eseul (eseul despre particularitățile unui text), cu fiecare paragraf pe un rând nou.')
        eseu = gr.Textbox(label='Eseul tău', lines=14,
                          placeholder='Poemul „Luceafărul” de Mihai Eminescu a apărut în 1883...')
        buton_v = gr.Button('Verifică', variant='primary')
        raport = gr.Markdown()
        tabel = gr.Dataframe(headers=['nr', 'ce acoperă', 'începutul paragrafului'],
                             label='Fiecare paragraf', wrap=True)
        buton_v.click(barem.verifica, inputs=eseu, outputs=[tabel, raport])


    with gr.Tab('Text argumentativ'):
        gr.Markdown('Lipește textul suport (fragmentul de la Subiectul I) și textul tău argumentativ.')
        fragment = gr.Textbox(label='Textul suport (copy-paste din subiect)', lines=8)
        text_arg = gr.Textbox(label='Textul tău argumentativ', lines=10)
        buton_a = gr.Button('Verifică', variant='primary')
        raport_a = gr.Markdown()
        tabel_a = gr.Dataframe(headers=['nr', 'ce este', 'începutul paragrafului'],
                               label='Fiecare paragraf', wrap=True)
        buton_a.click(barem_arg.verifica_argumentativ, inputs=[text_arg, fragment],
                      outputs=[tabel_a, raport_a])


    with gr.Tab('Generează text'):
        if model is None:
            gr.Markdown('Nu există `model.pt`. Rulează întâi `python train.py`, apoi repornește aplicația.')
        with gr.Row():
            with gr.Column():
                start = gr.Textbox(label='Început de text', placeholder='ROMEO:', lines=3)
                lungime = gr.Slider(50, 1000, value=300, step=10, label='Câte caractere să genereze')
                temperatura = gr.Slider(0.1, 2.0, value=0.8, step=0.1,
                                        label='Creativitate (temperature): mic = sigur, mare = creativ')
                top_k = gr.Slider(0, 65, value=0, step=1,
                                  label='top-k: alege doar dintre cele mai probabile k caractere (0 = toate)')
                buton_g = gr.Button('Generează', variant='primary')
            with gr.Column():
                rezultat = gr.Textbox(label='Text generat', lines=18)
                nota = gr.Markdown()
        buton_g.click(genereaza, inputs=[start, lungime, temperatura, top_k], outputs=[rezultat, nota])

if __name__ == '__main__':
    demo.launch(share='--share' in sys.argv)
