# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# Ricardo Bigolin Lanfredi - 2023-14-12
# Description:
# Script used to check for anonymization of generated location expressions

import torch.nn.functional as F
import argparse
import pandas as pd
import re
from collections import OrderedDict
from functools import reduce

check_expressions = [
'between thing_ pm and # : thing_', #
'between thing_ am thing_ and # : thing_', #
'between thing_ am thing_ and #', #
'between thing_ and #', #
'between # and #', #
'inferiorly on the right between thing_ and #', #
'bilaterally between thing_ and #',
'right mid lung between thing_ and #',
'between thing_ and thing_ am # thing_', #
'nasogastric tube also appearing between thing_ and # : thing_', #
'right pic catheter with tip appearing between thing_ am and # : thing_',
'between thing_ pm thing_ and thing_ am # thing_',
'also appearing between thing_ and # : thing_',
'mid lung zones developing between thing_ and #',
'between # and thing_',
] 

allowed_words = set()

# Open and read the file
with open("labeler_src/allowed_words.txt", 'r') as file:
    for line in file:
        # Remove leading and trailing whitespace, and convert to lowercase (if needed)
        word = line.strip().lower()
        
        # Add the word to the set
        allowed_words.add(word)

for single_file in [
                    './experiment_test_annotations/nih_human_annotations.csv'
                    './new_dataset_annotations/nih_llm_annotations_test.csv'
                    './new_dataset_annotations/nih_llm_annotations_train.csv'
                    './new_dataset_annotations/nih_llm_annotations_val.csv']:
    all_words = set()
    all_locations_table = pd.read_csv(single_file)
    all_locations_table = all_locations_table[all_locations_table['type_annotation']=='location']

    abnormality_names = []
    for column_name in all_locations_table.columns:
        # print(column_name)
        if column_name not in ['report', 'type_annotation', 'subjectid_studyid']:
            abnormality_names.append(column_name)
            for location_expressions in all_locations_table[all_locations_table[column_name]!=-1][column_name].values:
                # print(location_expressions)
                if location_expressions!='-1' and location_expressions==location_expressions and location_expressions!=1:
                    list_of_expressions = location_expressions.split(';')
                    
                    for expression in list_of_expressions:
                        original_expression = expression
                        expression = expression.replace('(','').replace(')','').replace('.','').replace('?','').replace('  ',' ').replace('  ',' ').strip()
                        
                        for number in range(10):
                            expression = expression.replace(str(number),'#')
                        for number in ['third','fourth','fifth','sixth','seventh','eighth','ninth','tenth','eleventh']:
                            expression = expression.replace(str(number),'#th')
                        # if expression in check_expressions:
                        #     all_words.add(original_expression)
                        list_of_words = expression.split(' ')
                        for word in list_of_words:
                            if word not in allowed_words:
                                all_words.add(expression)
                                break
    # for word in all_words:
    #     indices = reduce(lambda a, b: a | b, 
    #                          [all_locations_table[col].str.contains(word) for col in abnormality_names])
    #     if len(all_locations_table[indices].values)>0:
    #         print(word, all_locations_table[indices].values)
    #     print()
    #     print()
df = pd.DataFrame({'Column': list(all_words)})

# Write to CSV
df.to_csv('./location_words_check_anonymization.csv', index=False)
            