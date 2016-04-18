import csv, pickle
from nltk.corpus import stopwords

stwords = stopwords.words('english')

radlex_terms_1word = []
radlex_terms_2word = []


with open('./radlex/Radlex.csv') as csvfile:
  lexreader = csv.reader(csvfile, delimiter=',')
  for row in lexreader:

    for rowi in row[0].split():
      try:
        rowia = rowi.rstrip().encode('ascii', 'ignore').lower()
      except:
        continue
      if rowia not in radlex_terms_1word and rowia not in stwords\
      and not rowia.isdigit():
        radlex_terms_1word.append(rowia)

    row0 = row[0].split()
    for rid in range(len(row0)-1):
      rowi2 = ' '.join(row0[rid:rid+2])
      try:
        rowi2a = rowi2.rstrip().encode('ascii', 'ignore').lower()
      except:
        continue
      if rowi2a not in radlex_terms_2word:
        stwordext = False
        for rowi2ai in rowi2a.split():
          if rowi2ai in stwords:
            stwordext = True
        if stwordext:
          pass
        else:
          radlex_terms_2word.append(rowi2a)          


radlex_terms = {'1word': radlex_terms_1word, '2word': radlex_terms_2word}

with open('radlex_terms.pkl', 'wb') as handle:
  pickle.dump(radlex_terms, handle)
