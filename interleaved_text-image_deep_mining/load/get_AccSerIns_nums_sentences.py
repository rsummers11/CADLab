import re, csv, math, mysql.connector
import nltk.data
import sys


def import_into():
  cnx = mysql.connector.connect(host='mysql_server', database='dbname',
    user='user', password='****')
  cursor = cnx.cursor()


  cursor.execute('SELECT accnum, report FROM ris_reports')
  reportsRes = cursor.fetchall()


  non_matching_tuples = []


  for i in range(len(reportsRes)):
    print str(i) + '/' + str(len(reportsRes))
    report = reportsRes[i][1].encode('ascii', 'ignore').lower()
    accnum = reportsRes[i][0]

    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sents = sent_detector.tokenize(report.strip())
    for senti in range(len(sents)):
      sent = sents[senti]
      
      accSerInsTriples = []
      if ('series' in sent or 'serieses' in sent)\
        and ('image' in sent or 'images' in sent):

        sernums_extracted = []
        insnums_extracted = []
        words = []
        words0 = sent.split()
        for wsi0 in range(len(words0)):
          hyphenInTheWord = '-' in words0[wsi0]
          words0is = words0[wsi0].split('-')
          words0is2 = []
          for words0isi in words0is:
            words0isi2 = re.sub(r'[\W_]+', '', words0isi)
            words0is2.append(words0isi2)
          if (words0[wsi0-1]=='image' or words0[wsi0-1]=='images')\
            and words0is2[0].isdigit() and words0is2[-1].isdigit()\
            and hyphenInTheWord:
            words.append(words0[wsi0])
          elif (words0[wsi0-2]=='image' or words0[wsi0-2]=='images')\
            and words0is2[0].isdigit() and words0is2[-1].isdigit()\
            and hyphenInTheWord:
            words.append(words0[wsi0])
          else:
            for words0isi in words0is:
              words.append(words0isi)
          
        sent2 = []
        sent2_and_to_be_excluded_from_idxes = []
        for j in range(len(words)):
          if '-' in words[j]:
            wordl = words[j]
          else:
            wordl = re.sub(r'[\W_]+', '', words[j])
          
          if len(words)>j+1 and wordl[0:6]=='series'\
            and re.sub(r'[\W_]+', '', words[j+1]).isdigit():
            if '/' in words[j+1]:
              sernums_extracted1 = words[j+1].split('/')
              for sernums_extracted1i in sernums_extracted1:
                if re.sub('[^0-9]', '', sernums_extracted1i).isdigit():
                  sernums_extracted.append(int(re.sub('[^0-9]', '', sernums_extracted1i)))
            else:
              sernums_extracted.append(int(re.sub('[^0-9]', '', words[j+1])))
          elif len(words)>j+1 and wordl=='image'\
            and re.sub(r'[\W_]+', '', words[j+1]).isdigit()\
            and '-' not in words[j+1]:
            if '/' in words[j+1]:
              insnums_extracted1 = words[j+1].split('/')
              for insnums_extracted1i in insnums_extracted1:
                if insnums_extracted1i.isdigit():
                  insnums_extracted.append(int(insnums_extracted1i))
            elif ',' in words[j+1]:
              insnums_extracted1 = words[j+1].split(',')
              for insnums_extracted1i in insnums_extracted1:
                if re.sub('[^0-9]', '', insnums_extracted1i).isdigit():
                  insnums_extracted.append(int(re.sub('[^0-9]', '', insnums_extracted1i)))
            else:
              insnums_extracted.append(int(re.sub('[^0-9]', '', words[j+1])))
          elif wordl=='images' or\
            (wordl=='image' and len(words)>j+1 and '-' in words[j+1]):
            insnum_beg = 0
            insnum_end = 0
            if len(words)>j+1 and '-' in words[j+1]:
              insnums_extracted1 = re.split('-|,|;|:|\.|~',words[j+1])
              if re.sub('[^0-9]', '', insnums_extracted1[0]).isdigit()\
                and re.sub('[^0-9]', '', insnums_extracted1[1]).isdigit():
                insnum_beg = int(re.sub('[^0-9]', '', insnums_extracted1[0]))
                insnum_end = int(re.sub('[^0-9]', '', insnums_extracted1[1]))

                if insnum_end<insnum_beg and insnum_end<10:
                  insnum_end = int(math.floor(insnum_beg/10.0)*10 + insnum_end)
                elif insnum_end<insnum_beg and insnum_end>=10 and insnum_end<100:
                  insnum_end = int(math.floor(insnum_beg/100.0)*100 + insnum_end)
                else:
                  pass
              
              if insnum_end>insnum_beg:
                pass
              else:
                insnum_tmp = insnum_beg
                insnum_beg = insnum_end
                insnum_end = insnum_tmp
            
              if insnum_end>0:
                insnums_extracted = range(insnum_beg, insnum_end+1)
              
              if not insnums_extracted1[2:]:
                pass
              else:
                for insnums_extracted1resti in insnums_extracted1[2:]:
                  if insnums_extracted1resti.isdigit():
                    insnums_extracted.append(int(insnums_extracted1resti))

              if len(words)>j+2 and re.sub('[^0-9]', '', words[j+2]).isdigit():
                if '-' in words[j+2]:
                  possibleNextNumbers = words[j+2].split('-')
                  if len(possibleNextNumbers)>1\
                    and int(re.sub('[^0-9]', '', possibleNextNumbers[1])) > int(re.sub('[^0-9]', '', possibleNextNumbers[0])):
                    for possibleNextNumbersi in range(int(re.sub('[^0-9]', '', possibleNextNumbers[0])),int(re.sub('[^0-9]', '', possibleNextNumbers[1]))+1):
                      insnums_extracted.append(possibleNextNumbersi)
                else:
                  insnums_extracted.append(int(re.sub('[^0-9]', '', words[j+2])))

            elif len(words)>j+2 and words[j+2]=='and'\
              and re.sub('[^0-9]', '', words[j+1]).isdigit()\
              and re.sub('[^0-9]', '', words[j+3]).isdigit():
              insnum_beg = int(re.sub('[^0-9]', '', words[j+1]))
              insnum_end = int(re.sub('[^0-9]', '', words[j+3]))
              sent2_and_to_be_excluded_from_idxes.append(j+2)
            else:
              pass
          else:
            pass

          sent_current = re.sub(r':(?=..(?<!\d:\d\d))|[^a-zA-Z0-9 ](?<!:)',
            ' ',sents[senti])
          if len(sents)>senti+1:
            sent_following = re.sub(r':(?=..(?<!\d:\d\d))|[^a-zA-Z0-9 ](?<!:)',
              ' ',sents[senti+1])
          else:
            sent_following = ''
          if senti-1>0:
            sent_previous = re.sub(r':(?=..(?<!\d:\d\d))|[^a-zA-Z0-9 ](?<!:)',
              ' ',sents[senti-1])
          else:
            sent_previous = ''

          if (not sernums_extracted or not insnums_extracted)\
            or (accnum, sernums_extracted, insnums_extracted) in accSerInsTriples:
            pass
          else:
            qstr = 'INSERT INTO ris_im_mention_sentences (accnum, '+\
              'sernums,insnums,sentence,sentence_previous,sentence_following) '+\
              'VALUES ("' + accnum + '", "' +\
              str(sernums_extracted).lstrip('[').rstrip(']') + '", "' +\
              str(insnums_extracted).lstrip('[').rstrip(']') + '", "' +\
              sent_current + '", "' + sent_previous + '", "' +\
              sent_previous + '")'
                    
            cursor.execute(qstr)
            accSerInsTriples.append((accnum, sernums_extracted, insnums_extracted))

  cursor.close()


def sentences_to_csv_previous_following():
  cnx = mysql.connector.connect(host='mysql_server', database='dbname',
    user='user', password='****')
  cursor = cnx.cursor()

  qstr = 'SELECT id, sentence_previous, sentence, sentence_following '+\
    ' FROM ris_im_mention_sentences'
  cursor.execute(qstr)

  csvf = '___csv file to store___'
  with open(csvf, 'wb') as csv_file:
    csv_writer = csv.writer(csv_file)

    for row in cursor:
      if not row[1]:
        pass
      else:
        sent1 = ' ' + row[1].lstrip(' ').rstrip(' ').encode('ascii', 'ignore') + ' ' +\
          ' ' + row[2].lstrip(' ').rstrip(' ').encode('ascii', 'ignore') + ' '+\
          ' ' + row[3].lstrip(' ').rstrip(' ').encode('ascii', 'ignore') + ' '
        sent2l = []
        for sent1i in sent1.split():
          if not sent1i.isdigit():
            sent2l.append(sent1i)
        sent2 = ' '.join(sent2l)
        row1 = (row[0], sent2)
        csv_writer.writerow(row1)

  cursor.close()


def sentences_to_csv():
  cnx = mysql.connector.connect(host='mysql_server', database='dbname',
    user='user', password='****')
  cursor = cnx.cursor()

  qstr = 'SELECT id, sentence FROM ris_im_mention_sentences'
  cursor.execute(qstr)

  csvf = '___csv file to store___'
  with open(csvf, 'wb') as csv_file:
    csv_writer = csv.writer(csv_file)

    for row in cursor:
      if not row[1]:
        pass
      else:
        sent1 = ' ' + row[1].lstrip(' ').rstrip(' ').encode('ascii', 'ignore') + ' '
        sent2l = []
        for sent1i in sent1.split():
          if not sent1i.isdigit():
            sent2l.append(sent1i)
        sent2 = ' '.join(sent2l)
        row1 = (row[0], sent2)
        csv_writer.writerow(row1)

  cursor.close()


if __name__ == "__main__":
  sentences_to_csv()
  sentences_to_csv_previous_following()
  