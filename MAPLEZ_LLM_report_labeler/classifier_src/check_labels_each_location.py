# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# Ricardo Bigolin Lanfredi - 2023-14-12
# This script takes as input the LLM lables, 
# report_labels_mimic_0.0_llama2_location_redone_1_30.csv
# and outputs to stdout
# Description:
# File used to get a list of locations keywords that appear in at 
# least 2% of positive keywords

import torch.nn.functional as F
import argparse
import pandas as pd
import re
from collections import OrderedDict
from list_labels import str_labels_location as list_of_words
from list_labels import list_of_replacements_labels_location as list_of_replacements

parser = argparse.ArgumentParser()
parser.add_argument('--single_file', type=str, default = './new_dataset_annotations/mimic_llm_annotations.csv')

def findWholeWord(w):
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

args = parser.parse_args()
all_locations_table = pd.read_csv(args.single_file)
all_locations_table = all_locations_table[all_locations_table['type_annotation']=='location']

for column_name in all_locations_table.columns:
    print(column_name)
    if column_name not in ['report', 'type_annotation', 'subjectid_studyid']:
        dict_all = OrderedDict()
        total_words = 0
        for word in list_of_words:
            dict_all[word] = 0
        
        for location_expressions in all_locations_table[all_locations_table[column_name]!=-1][column_name].values:
            sentence = ', '.join(location_expressions.split(';'))
            if sentence!='-1':
                word_found = False
                for word in set(list_of_words + list(list_of_replacements.keys())):
                    
                    if findWholeWord(word)(sentence):
                        if word in list_of_replacements.keys():
                            for replacement_word in list_of_replacements[word]:
                                dict_all[replacement_word] += 1
                        else:
                            dict_all[word] += 1
                        word_found = True
                if word_found:
                    total_words += 1
        for word in list_of_words:
            dict_all[word] /= total_words
        for word in list_of_words:
            if dict_all[word]>0.02:
                print(word, dict_all[word])