# Ghid de etichetare — BacHelper

Fișierul de date: `paragrafe_etichetate.csv` (se deschide în Excel; coloanele sunt separate prin `;`).
Fiecare rând = un paragraf dintr-un eseu.

## Coloane

| coloană | ce scrii |
|---|---|
| eseu_id | un număr pentru fiecare eseu (1, 2, 3 ...); toate paragrafele aceluiași eseu au același număr |
| opera | titlul operei (de ex. Luceafărul) |
| autor | autorul operei |
| tip_eseu | deocamdată mereu `particularitati` |
| nr_paragraf | al câtelea paragraf e în eseu (1, 2, 3 ...) |
| eticheta | una dintre etichetele de mai jos, scrisă exact așa |
| observatii | opțional: de ce ai ales eticheta, dacă nu e evident |
| sursa | `gasit` = scris de un om (eseu, fișă, culegere); `generat_AI` = generat cu ChatGPT sau alt AI. Eseurile AI se folosesc doar la antrenare, nu la evaluare |
| text | textul paragrafului |

## Etichete (după baremul Subiectului al III-lea)

| etichetă | când o folosești |
|---|---|
| `introducere` | paragrafele de la început: opera, autorul, anul, contextul apariției, definiții generale ale curentului sau speciei (fără exemple din text) |
| `incadrare` | o trăsătură, ilustrată în text, care încadrează opera într-o perioadă, un curent (romantism, realism...), o orientare tematică sau o specie (basm cult, nuvelă...) |
| `tema` | tema operei și viziunea despre lume |
| `secventa` | comentarea unei scene / secvențe / episod (inclusiv paragrafele care continuă aceeași secvență) |
| `element_structura` | titlu, incipit, final, conflict, perspectivă narativă, personaj, relații temporale și spațiale, motive, figuri de stil, limbaj |
| `concluzie` | paragraful de încheiere |
| `altele` | ce nu răspunde niciunei cerințe (biografie lungă, idei generale fără legătură) |

## Reguli

1. **O singură etichetă pe paragraf.** Dacă un paragraf face două lucruri, alege-l pe cel principal și notează-l pe celălalt la observații.
2. **Paragrafele care continuă o idee** primesc aceeași etichetă ca paragraful în care a început ideea.
3. **Eseuri diferite, structuri diferite.** Ajută mult să ai eseuri scrise de oameni diferiți, pe opere diferite, și nu doar eseuri „perfecte”: și eseuri cărora le lipsește ceva.
4. **Ținta:** cel puțin 30 de eseuri (~300 de paragrafe). Mai mult e și mai bine.

## Drepturi de autor

Comentariile de pe site-uri și din culegeri au autori. Folosește-le doar local, pentru antrenare,
și **nu urca fișierul de date pe GitHub**. În repo pune doar codul și câteva exemple scrise de tine.
