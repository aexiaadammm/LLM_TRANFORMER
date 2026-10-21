# BacHelper

Aplicație pentru elevii de clasa a XII-a care se pregătesc de bacalaureatul la limba română.
În loc să memoreze comentarii, elevul își verifică propriul text după barem și vede ce îi lipsește.

Proiectul conține și un model de limbaj de tip **Transformer implementat de la zero în PyTorch**,
folosit în două feluri: ca generator de text (decoder, ca GPT) și ca clasificator de paragrafe (encoder, ca BERT).

## Funcționalități

| tab | ce face |
|---|---|
| **Verifică eseul** | eseul despre particularitățile unui text (Subiectul al III-lea): etichetează fiecare paragraf (introducere, încadrare, temă, secvență, element de structură, concluzie) și spune ce cerințe din barem lipsesc |
| **Text argumentativ** | Subiectul I B: găsește opinia, argumentele și concluzia, numără conectorii și cuvintele și verifică dacă argumentele se raportează la textul suport lipit de elev |
| **Generează text** | modelul GPT antrenat de la zero continuă un început de text, caracter cu caracter |

<!-- adaugă aici o captură de ecran: docs/captura.png -->

## Cum funcționează

```
text ──► caractere ──► embedding-uri (token + poziție)
                         │
                Block × n_layer
                  x = x + Atenție multi-head(LayerNorm(x))
                  x = x + Feed-forward(LayerNorm(x))
                         │
          ┌──────────────┴──────────────┐
   decoder (mască cauzală)        encoder (fără mască)
   → caracterul următor           → media vectorilor → eticheta paragrafului
   GPTLanguageModel               TransformerClassifier
```

| fișier | rol |
|---|---|
| `model.py` | arhitectura: `Head`, `MultiHeadAttention`, `FeedForward`, `Block`, `GPTLanguageModel`, `TransformerClassifier` |
| `train.py` | antrenează generatorul și îl salvează în `model.pt` |
| `train_clasificator.py` | antrenează clasificatorul de paragrafe, îl evaluează și îl salvează în `clasificator.pt` |
| `prelucrare.py` | pregătirea textului, împărțirea în paragrafe, regulile pe cuvinte-cheie |
| `barem.py` | verificarea eseului după barem |
| `barem_arg.py` | verificarea textului argumentativ după barem |
| `app.py` | interfața web (Gradio) |
| `main.py` | varianta inițială, totul într-un singur fișier, scrisă urmărind tutorialul lui Andrej Karpathy |

## Evaluare

Clasificatorul e evaluat pe eseuri pe care nu le-a văzut (*leave-one-essay-out*): pentru fiecare eseu,
modelul se antrenează pe celelalte și e testat pe el. Eseurile generate cu AI sunt folosite doar la antrenare,
nu și la test.

| metodă | paragrafe etichetate corect (9 eseuri, 144 de paragrafe) |
|---|---|
| reguli pe cuvinte-cheie (baseline) | 69% |
| Transformer encoder antrenat de la zero | 41% |

Cu atât de puține date, Transformer-ul memorează exemplele în loc să generalizeze (overfitting), așa că
aplicația folosește automat metoda cu rezultatul mai bun. Când setul de date crește și Transformer-ul
depășește regulile, aplicația trece singură pe el. Cu 24 de paragrafe în plus, scorul lui a crescut de la 35% la 41%.

## Cum se rulează

```
pip install -r requirements.txt
python train.py               # generatorul (descarcă singur textul de antrenare)
python train_clasificator.py  # clasificatorul (are nevoie de date/paragrafe_etichetate.csv)
python app.py                 # interfața: http://127.0.0.1:7860
```

Verificarea textului argumentativ și a eseului funcționează și fără `clasificator.pt`, cu regulile pe cuvinte-cheie.

## Date

Eseurile folosite la antrenare provin din culegeri și materiale de pregătire și au drepturi de autor,
de aceea **nu sunt incluse** în repository. Formatul datelor și regulile de etichetare sunt descrise în
`date/ghid_etichetare.md`, iar `date/exemplu_argumentative.csv` arată formatul pentru textele argumentative.

## Limite

- Aplicația verifică **structura** și prezența cerințelor din barem, nu și calitatea argumentării sau corectitudinea gramaticală.
- Fiecare paragraf primește o singură etichetă; un paragraf care conține mai multe cerințe (de ex. tema și ambele secvențe) e recunoscut doar parțial.
- Raportarea la textul suport e verificată prin cuvinte comune și trimiteri explicite; parafrazele nu sunt recunoscute.

## Planuri

- mai multe eseuri și texte argumentative, scrise de elevi
- pre-antrenarea Transformer-ului pe comentarii neetichetate, apoi fine-tuning pe paragrafele etichetate
- etichetarea la nivel de frază
- generator antrenat pe texte în limba română, cu tokenizare pe subcuvinte

## Resurse

- Andrej Karpathy, *Let's build GPT: from scratch, in code, spelled out*
- Vaswani et al., *Attention Is All You Need* (2017)
