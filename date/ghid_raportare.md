# Ghid de etichetare: raportarea la textul suport

Scopul: un model care primește **un fragment** (textul suport de la Subiectul I) și **un argument**
(un paragraf dintr-un text argumentativ) și spune dacă argumentul **folosește textul suport**.
Baremul Subiectului I B dă 3 puncte pentru valorificarea textului și doar 1 punct pentru simpla citare.

Sunt două fișiere (se deschid în Excel; coloanele sunt separate prin `;`):

## 1. `fragmente.csv`: textele suport

| coloană | ce scrii |
|---|---|
| fragment_id | un număr pentru fiecare fragment (1, 2, 3 ...) |
| sursa | `gasit` = dintr-o variantă de bac sau culegere; `exemplu` = rândurile model din fișier |
| text | textul fragmentului, între ghilimele |

## 2. `raportare.csv`: argumentele

Fiecare rând = un paragraf-argument, legat de un fragment prin `fragment_id`.

| coloană | ce scrii |
|---|---|
| fragment_id | numărul fragmentului la care a fost scris argumentul |
| legat | `1` = argumentul folosește textul suport; `0` = nu îl folosește |
| sursa | `gasit` = scris de un om (model de rezolvare, tu, colegi); `generat_AI` = generat cu un AI (doar pentru antrenare, nu pentru evaluare); `exemplu` = rândurile model |
| observatii | opțional: de ce ai ales eticheta |
| argument | textul paragrafului, între ghilimele |

## Când pui 1 și când pui 0

**`legat = 1`**: argumentul preia o idee, o situație sau un exemplu din fragment și o dezvoltă.
- și cu cuvintele textului („așa cum arată textul, ...”)
- și **parafrazat**, cu alte cuvinte; acestea sunt cele mai valoroase exemple, pentru că regula actuală nu le recunoaște

**`legat = 0`**: argumentul e pe **aceeași temă**, dar nu folosește nimic din fragment
(un exemplu personal, o idee generală, o informație din altă parte).
Acestea sunt „cazurile grele” și sunt foarte importante: fără ele, modelul ar învăța doar „e despre același subiect”.

Nu e nevoie să scrii argumente fără nicio legătură (de ex. despre sport la un fragment despre lectură):
scriptul le creează singur, punând un argument lângă alt fragment.

## Câte exemple

- Țintă pentru început: **5–10 fragmente** și **30–50 de argumente**, cam jumătate `1` și jumătate `0`.
- Pentru fiecare fragment încearcă să ai cel puțin un `1` cu cuvintele textului, un `1` parafrazat și un `0` pe aceeași temă.

## Atenție

- Un text care conține `;` sau rânduri noi trebuie pus **între ghilimele** (Excel o face singur când salvezi ca CSV).
- Salvează ca **CSV UTF-8**, ca să rămână diacriticele.
- Fragmentele din variantele de bac pot avea drepturi de autor: fișierele rămân doar pe calculator
  (sunt excluse din GitHub prin `.gitignore`); pe GitHub ajunge doar acest ghid.
