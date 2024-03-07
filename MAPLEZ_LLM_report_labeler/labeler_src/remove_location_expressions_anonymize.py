# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# Ricardo Bigolin Lanfredi - 2023-13-12
# The input of this script is 
# './experiment_test_annotations/nih_human_annotations.csv'
# './new_dataset_annotations/nih_llm_annotations_test.csv'
# './new_dataset_annotations/nih_llm_annotations_train.csv'
# './new_dataset_annotations/nih_llm_annotations_val.csv', i.e. the files containing
# the LLM or human location annotations for NIH ChestXray14 reports
# And the Output is 
# 'val_nih_labeled_id_converted_anonymized.csv',
# 'train_nih_labeled_id_converted_anonymized.csv',
# 'test_nih_labeled_id_converted_anonymized.csv',
# 'sampled_nih_id_converted_anonymized.csv',
# where location expressions that might contain sensitive data 
# are removed
# Description:
# This script removes from the MAPLEZannotations
# for the NIH ChestXray14 dataset
#  location expressions that might contain sensitive data

import pandas as pd

check_expressions = [
'between thing_ pm and # : thing_',
'between thing_ am thing_ and # : thing_',
'between thing_ am thing_ and #',
'between thing_ and #',
'between # and #',
'inferiorly on the right between thing_ and #',
'bilaterally between thing_ and #',
'right mid lung between thing_ and #',
'between thing_ and thing_ am # thing_',
'nasogastric tube also appearing between thing_ and # : thing_',
'right pic catheter with tip appearing between thing_ am and # : thing_',
'between thing_ pm thing_ and thing_ am # thing_',
'also appearing between thing_ and # : thing_',
'mid lung zones developing between thing_ and #',
'between # and thing_',
] 

# Define the condition and modification
def modify_row(row):
    if row['type_annotation']=='location':
        for column_name in row.keys():
            if column_name not in ['report', 'type_annotation', 'subjectid_studyid']:
                if row[column_name]!=-1:
                    location_expressions = row[column_name]
                    if location_expressions!='-1' and location_expressions==location_expressions and location_expressions!=1:
                        list_of_expressions = location_expressions.split(';')
                        for expression in list_of_expressions[:]:
                            original_expression = expression
                            expression = expression.replace('(','').replace(')','').replace('.','').replace('?','').replace('  ',' ').replace('  ',' ').strip()
                            
                            for number in range(10):
                                expression = expression.replace(str(number),'#')
                            for number in ['third','fourth','fifth','sixth','seventh','eighth','ninth','tenth','eleventh']:
                                expression = expression.replace(str(number),'#th')
                            remove_expression = False
                            if expression in check_expressions:
                                remove_expression = True
                            if not remove_expression:
                                assert(not (('between thing_ and' in expression and '#' in expression) or 'between # and' in expression))
                            if remove_expression:
                                list_of_expressions.remove(original_expression)
                        row[column_name] = ';'.join(list_of_expressions)
    return row
for single_file in [
                    './experiment_test_annotations/nih_human_annotations.csv'
                    './new_dataset_annotations/nih_llm_annotations_test.csv'
                    './new_dataset_annotations/nih_llm_annotations_train.csv'
                    './new_dataset_annotations/nih_llm_annotations_val.csv']:
    # Apply the modification
    # Load the CSV file
    df = pd.read_csv(single_file)
    df = df.apply(modify_row, axis=1)
    # Save the modified DataFrame back to CSV
    df.to_csv('./'+single_file.replace('.csv','') + '_anonymized.csv', index=False)

