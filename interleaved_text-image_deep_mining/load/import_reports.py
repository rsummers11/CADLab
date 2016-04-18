import re, os, sys
import mysql.connector



def remove_patient_info(line):
  if line.lower().find('name:')>0 and line.lower().find('patient id')>0:
    line = line.replace(
      line[line.lower().find('name:'):line.lower().find('patient id:')+20],'')
  elif line.lower().find('name:')>0 and line.lower().find('patient id:')<0:
    line = line.replace(
      line[line.lower().find('name:'):line.lower().find('name:')+20],'')
  elif line.lower().find('patient id:')>0 and line.lower().find('name:')<0:
    line = line.replace(
      line[line.lower().find('patient id:'):line.lower().find('patient id:')+20],'')

  return line


def import1():

  ostype = sys.platform
    datasource = os.path.join('___path to the directory containing report files___')  

  f1 = open(os.path.join(datasource, '___csv stored reports___'))
  lines = f1.readlines()
  f1.close()


  re1_1 = '(\\d+)'
  re1_2 = '(,)'
  rg1 = re.compile(re1_1+re1_2, re.IGNORECASE|re.DOTALL)


  re2_1='.*?' # Non-greedy match on filler
  re2_2='(")' # Any Single Character 1
  re2_3='((?:[a-z][a-z0-9_]*))' # Variable Name 1
  re2_4='(")' # Any Single Character 2
  rg2 = re.compile(re2_1+re2_2+re2_3+re2_4,re.IGNORECASE|re.DOTALL)  


  re3_1='(\\d+)'  # Integer Number 1
  re3_2='(,)' # Any Single Character 1
  re3_3='(".*?")' # Double Quote String 1
  re3_4='(,)' # Any Single Character 2
  re3_5='(")' # Any Single Character 3
  re3_6='(.*)'
  re3_7='"'
  rg3 = re.compile(re3_1+re3_2+re3_3+re3_4+re3_5+re3_6+re3_7,re.IGNORECASE|re.DOTALL)


  cnx = mysql.connector.connect(host='mysql_server', database='dbname',
    user='user', password='****')
  cursor = cnx.cursor()

  for line in lines:

    line = line.rstrip('\n').rstrip('\r')

    m2 = rg2.search(line)
    accnum = m2.group(2)

    m3 = rg3.search(line)
    report = re.sub(' +', ' ', m3.group(6))

    # triple checking
    for i in range(5):
      report = remove_patient_info(report)

    qstr = "INSERT INTO ris_reports (accnum, source, report) VALUES " +\
      "('"+ str(accnum) + "', '" + "accnum_reports" + "', '" + report + "')"

    cursor.execute(qstr)


  cnx.commit()
  cnx.close()


def import2():

  ostype = sys.platform
  datasource = os.path.join('___path to the directory containing report files___')
  
  f2 = open(os.path.join(datasource, '___txt stored (using simple dump) reports___'))
  lines = f2.readlines()
  f2.close()


  re1_1='(Dictating Physician:|Dictating Radiologist:)'
  re1_2='.*?' # Non-greedy match on filler
  rg1 = re.compile(re1_1+re1_2,re.IGNORECASE|re.DOTALL)


  re2_1='.*?'
  re2_2='((?:[a-z][a-z]*[0-9]+[a-z0-9]*))'
  rg2 = re.compile(re2_1+re2_2,re.IGNORECASE|re.DOTALL)


  re3_1='(Patient:)'
  re3_2='.*?' # Non-greedy match on filler
  rg3 = re.compile(re3_1+re3_2,re.IGNORECASE|re.DOTALL)


  re4_1='(Name:)'
  re4_2='.*?' # Non-greedy match on filler
  rg4 = re.compile(re4_1+re4_2,re.IGNORECASE|re.DOTALL)


  re5_1='(Patient ID:)'
  re5_2='.*?' # Non-greedy match on filler
  rg5 = re.compile(re5_1+re5_2,re.IGNORECASE|re.DOTALL)


  start = False
  report = ''


  cnx = mysql.connector.connect(host='mysql_server', database='dbname',
    user='user', password='****')
  cursor = cnx.cursor()


  for line in lines:
      
    m1 = rg1.search(line)
    if m1:
      start = False
      if report:
        report = re.sub(' +', ' ', report)
        ## insert into the database
        qstr = "INSERT INTO ris_reports (accnum, source, report) VALUES " +\
          "('"+ str(accnum) + "', '" + "dictating_following" + "', '" + report + "')"
        cursor.execute(qstr)
        ###
        report = ''

    if not start:
      m2 = rg2.search(line)
      if m2 and len(m2.group(1))>=8 and len(m2.group(1))<=15:
        accnum = m2.group(1)
        
    if start and line!='\n':
      m4 = rg4.search(line)
      m5 = rg5.search(line)

      if (not m4) and (not m5):
        # triple checking
        for i in range(5):
          line = remove_patient_info(line)

        report = report + line.rstrip('\n') + ' '

    m3 = rg3.search(line)
    if m3:
      start = True


  cnx.commit()
  cnx.close()


if __name__ == "__main__":
  import1()
  import2()
