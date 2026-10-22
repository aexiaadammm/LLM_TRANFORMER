from sentence_transformers import SentenceTransformer, util

# prima rulare descarcă modelul (~470 MB) de pe Hugging Face; apoi îl ia din calculator
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

propozitii = [
    'Lectura ne învață empatie.',                          # 0
    'Cărțile ne ajută să înțelegem ce simt ceilalți.',
    'Lectura îmbogățește vocabularul.',
    'Fotbalul se joacă cu unsprezece jucători.',           # 3 n are legatura
]

vectori = model.encode(propozitii)
print(vectori.shape)

scoruri = util.cos_sim(vectori, vectori)   # similaritatea cosinus, fiecare cu fiecare
for i in range(1, len(propozitii)):
    print(f'{scoruri[0][i]:.2f}  {propozitii[0]}  <->  {propozitii[i]}')