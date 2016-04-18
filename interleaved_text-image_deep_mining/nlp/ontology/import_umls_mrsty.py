import csv, pickle, re

mrstys = ['T081', 'T029', 'T023', 'T080', 'T060', 'T061', 'T190',\
  'T130', 'T082', 'T079', 'T047', 'T033', 'T042']

for mrsty in mrstys:
  umls_mrsty_1word = []
  mrstyFnameW1 = './umls_mrsty/'+\
    'umls_' + mrsty.lower() + '_1word.csv'
  with open(mrstyFnameW1) as csvfile1:
    lexreader = csv.reader(csvfile1, delimiter=',')
    for row in lexreader:
      try:
        rowa = row[14].rstrip().encode('ascii', 'ignore').lower()
      except:
        continue

      if rowa not in umls_mrsty_1word:
        umls_mrsty_1word.append(rowa)

  umls_mrsty_2word = []
  mrstyFnameW2 = './umls_mrsty/'+\
    'umls_' + mrsty.lower() + '_2word.csv'
  with open(mrstyFnameW2) as csvfile2:
    lexreader = csv.reader(csvfile2, delimiter=',')
    for row in lexreader:
      try:
        rowa = row[14].rstrip().encode('ascii', 'ignore').lower()
      except:
        continue
        
      if rowa not in umls_mrsty_2word and len(rowa.split())==2:
        umls_mrsty_2word.append(rowa)


  umls_mrsty_terms = {'1word': umls_mrsty_1word, '2word': umls_mrsty_2word}

  pklFname = 'umls_mrsty_' + mrsty.lower() + '_terms.pkl'
  with open(pklFname, 'wb') as handle:
    pickle.dump(umls_mrsty_terms, handle)

