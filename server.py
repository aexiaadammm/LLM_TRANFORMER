# -*- coding: utf-8 -*-
"""
server.py — backend-ul aplicației web BacHelper.

Pornește un server (FastAPI, vine instalat odată cu Gradio) care:
  - trimite browserului pagina din frontend/index.html
  - primește textele de la pagină și întoarce analiza, ca JSON:
      POST /api/eseu          {text}                         -> analiza eseului (barem.py)
      POST /api/argumentativ  {text, fragment}               -> analiza textului argumentativ (barem_arg.py)
      POST /api/genereaza     {start, lungime, temperatura, top_k} -> text generat (generator.py)

Pornire:  python server.py     apoi deschizi http://127.0.0.1:8000
"""

import os

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

import barem
import barem_arg
import generator

DIR = os.path.dirname(os.path.abspath(__file__))
app = FastAPI(title='BacHelper')


class CerereEseu(BaseModel):
    text: str = ''


class CerereArgumentativ(BaseModel):
    text: str = ''
    fragment: str = ''


class CerereGenerare(BaseModel):
    start: str = ''
    lungime: int = 300
    temperatura: float = 0.8
    top_k: int = 0


@app.get('/')
def pagina():
    return FileResponse(os.path.join(DIR, 'frontend', 'index.html'))


@app.get('/api/stare')
def stare():
    """ce poate face serverul: metoda clasificatorului și dacă există generatorul"""
    return {'metoda_eseu': barem.metoda(), 'generator': generator.disponibil}


@app.post('/api/eseu')
def api_eseu(c: CerereEseu):
    a = barem.analizeaza(c.text)
    return a if a else {'eroare': 'Lipește un eseu, cu paragrafele despărțite prin rânduri noi.'}


@app.post('/api/argumentativ')
def api_argumentativ(c: CerereArgumentativ):
    a = barem_arg.analizeaza_argumentativ(c.text, c.fragment)
    return a if a else {'eroare': 'Lipește textul argumentativ, cu paragrafele pe rânduri separate.'}


@app.post('/api/genereaza')
def api_genereaza(c: CerereGenerare):
    if not generator.disponibil:
        return {'eroare': 'Nu există model.pt. Rulează întâi python train.py, apoi repornește serverul.'}
    text, ignorate = generator.genereaza(c.start, c.lungime, c.temperatura, c.top_k)
    return {'text': text, 'ignorate': ignorate}


if __name__ == '__main__':
    print('BacHelper pornește la http://127.0.0.1:8000  (oprești cu Ctrl+C)')
    uvicorn.run(app, host='127.0.0.1', port=8000)
