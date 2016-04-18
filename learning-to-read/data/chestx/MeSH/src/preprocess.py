import csv, random
import numpy as np

random.seed(123)


meshtop_list = []
with open('../original/openi.mesh.top') as fr:
  lines = fr.readlines()

  for line in lines:
    linel = line.split('|')

    for termi in linel[1:]:
      termi2 = termi.lstrip().rstrip().lower().replace('/','_').replace(', ','_').replace(' ','_')
      if termi2 not in meshtop_list:
        meshtop_list.append(termi2)

meshtop_onehot_list = []
with open('../original/openi.mesh.top') as fr:
  lines = fr.readlines()

  for line in lines:
    linel = line.split('|')

    onehot_list = []
    for termi in linel[1:]:
      termi2 = termi.lstrip().rstrip().lower().replace('/','_').replace(', ','_').replace(' ','_')
      if termi2 in meshtop_list:
        onehot_list.append(meshtop_list.index(termi2))

    onehot1 = np.zeros(len(meshtop_list), dtype=np.int)
    for onehoti in onehot_list:
      onehot1[onehoti] = 1
    onehot2 = map(str, onehot1)
    onehot = ''.join(onehot2)
    if onehot not in meshtop_onehot_list:
      meshtop_onehot_list.append(onehot)

##

meshtop_list_all = []
with open('../original/openi.mesh.top') as fr:
  lines = fr.readlines()

  i=0
  for line in lines:

    linel = line.split('|')
    fname = linel[0][1:].replace('/', '_')

    terms_list = []
    onehot_list = []
    for termi in linel[1:]:
      termi2 = termi.lstrip().rstrip().lower().replace('/','_').replace(', ','_').replace(' ','_')
      terms_list.append(termi2)
      if termi2 in meshtop_list:
        onehot_list.append(meshtop_list.index(termi2))
    terms = '|'.join(terms_list)

    onehot1 = np.zeros(len(meshtop_list), dtype=np.int)
    for onehoti in onehot_list:
      onehot1[onehoti] = 1
    onehot2 = map(str, onehot1)
    onehot = ''.join(onehot2)

    onehot_index = meshtop_onehot_list.index(onehot)

    meshtop_list_all.append([str(i), fname, line[len(fname)+2:].lstrip().rstrip(),\
      terms, str(onehot_index), onehot])
    i+=1

with open('../processed/openi.mesh.top.csv','w') as csvwf:
  csvw = csv.writer(csvwf, delimiter=',')
  csvw.writerow(['id', 'fname', 'mesh_top_original', 'mesh_top', 'onehot_index', 'onehot'])
  for meshtop_list_alli in meshtop_list_all:
    csvw.writerow(meshtop_list_alli)

####

mesh_list_all = []

with open('../original/openi.mesh') as fr:
  with open('../original/openi.mesh.top') as fr2:
    lines = fr.readlines()
    lines2 = fr2.readlines()

    i=0
    for line in lines:
      linel = line.split('|')
      fname = linel[0][1:].replace('/', '_')

      line2 = lines2[i]
      linel2 = line2.split('|')
      fname2 = linel2[0][1:].replace('/', '_')      

      terms_list_raw0 = []
      terms_list = []
      for termi22_ in linel2[1:]:
        termi22 = termi22_.lstrip().rstrip().lower()
        topterm = termi22.replace('/','_').replace(', ','_').replace(' ','_')

        for termi in linel[1:]:
          termi2_ = termi.lstrip().rstrip().lower()
          if termi22 in termi2_:
            termi2 = termi2_
            break

        if len(termi2)>len(termi22):          
          resterm0 = termi2[len(termi22)+1:]
          resterm0l = resterm0.split('/')
          resterm1l = []
          terms_list_raw0.append(topterm)
          for resterm0li in resterm0l:
            resterm0li2 = resterm0li.replace(', ','_').replace(' ','_')
            resterm1l.append(resterm0li2)
            terms_list_raw0.append(resterm0li2)
          resterm2 = '/'.join(resterm1l)
          termi3 = topterm + '/' + resterm2
        else:
          termi3 = topterm
          terms_list_raw0.append(termi3)
        terms_list.append(termi3)
      terms_raw = ' '.join(terms_list_raw0)
      terms_structured = '|'.join(terms_list)

      mesh_list_all.append([str(i), fname, meshtop_list_all[i][2], line[len(fname)+2:].lstrip().rstrip(),\
        meshtop_list_all[i][3], terms_structured, terms_raw, meshtop_list_all[i][4], meshtop_list_all[i][5]])
      i+=1

with open('../processed/openi.mesh.csv', 'w') as csvwf:
  csvw = csv.writer(csvwf, delimiter=',')
  csvw.writerow(['id', 'fname', 'mesh_top_original', 'mesh_original', 'mesh_top', 'mesh', 'terms_raw', 'onehot_index', 'onehot'])
  for mesh_list_alli in mesh_list_all:
    csvw.writerow(mesh_list_alli)
