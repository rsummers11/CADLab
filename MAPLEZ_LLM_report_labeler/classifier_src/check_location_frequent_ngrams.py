# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# Ricardo Bigolin Lanfredi - 2023-14-12
# This script takes as input the LLM lables, 
# report_labels_mimic_0.0_llama2_location_redone_1_30.csv
# and outputs to stdout
# Description:
# list the most frequent ngrams for location expressions

from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from transformers import AutoProcessor,AutoModel,AutoTokenizer
import torch.nn.functional as F
from torch import Tensor
import argparse
import pandas as pd
from collections import Counter
from nltk import ngrams
from sklearn.feature_extraction.text import CountVectorizer
import re
import collections
import numpy as np
import uuid
import h5py
from retry import retry

class Open(object):
    @retry((FileNotFoundError, IOError, OSError), delay=1, backoff=2, max_delay=10, tries=100)
    def __init__(self, file_name, method):
        self.file_obj = open(file_name, method)
    def __enter__(self):
        return self.file_obj
    def __exit__(self, type, value, traceback):
        self.file_obj.close()

@retry((FileNotFoundError, IOError, OSError), delay=1, backoff=2, max_delay=10, tries=100)
def get_lock(lock):
    lock.acquire()

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='a', choices = ['a','b','c','d','e','f'], help='')
parser.add_argument("--result_root", type=str, default = './', help='''''')
parser.add_argument('--single_file', type=str, default = './new_dataset_annotations/mimic_llm_annotations.csv')

def ngrams_per_line(n):
    def specific_ngram(doc):
        for ln in doc.split('\n'):
            terms = re.findall(r'\w{'+str(n)+',}', ln)
            shifted_lists = [terms[i:] for i in range(n)]
            for bigram in zip(*shifted_lists):
                yield ' '.join(bigram)
    return specific_ngram

args = parser.parse_args()
all_locations_table = pd.read_csv(args.single_file)
all_locations_table = all_locations_table[all_locations_table['type_annotation']=='location']
set_of_locations = set()
for column_name in all_locations_table.columns:
    # if column_name not in ['report', 'type_annotation', 'subjectid_studyid']:
    # if column_name in ['fracture']:
    if column_name in ['support device']:
        set_of_locations = set_of_locations.union(all_locations_table[all_locations_table[column_name]!=-1][column_name].values)

text = '\n'.join('\n'.join(set_of_locations).split(';'))
for n in range(1,4):
    cv = CountVectorizer(analyzer=ngrams_per_line(n))
    a = cv.fit_transform([text])
    ngrams_words = cv.get_feature_names_out()
    counts = a.toarray()[0]
    highest_counts = np.argsort(counts)[::-1]
    print(n)
    print(ngrams_words[highest_counts[:80]])
    print(counts[highest_counts[:80]])

