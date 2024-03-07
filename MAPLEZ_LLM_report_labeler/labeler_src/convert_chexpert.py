# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# Ricardo Bigolin Lanfredi - 2023-14-12
# The script takes the nih_chexpert_labeled.csv
# file as input and outputs to chexpert_nih_dataset_converted.csv
# Description:
# Script to convert the cheXpert labels for the NIH ChestXray14 dataset 
# to the format used by our labeler

import json
import pandas as pd

columns_to_use = ['pneumothorax','pleural effusion','fracture', 'cardiomegaly','lung opacity','consolidation','lung edema','atelectasis']

columns_to_join_preds = {'cardiomegaly':['Cardiomegaly'],
                        'lung edema':['Edema'],
                        'lung opacity':['Lung Opacity', 'Consolidation', 'Edema', 'Atelectasis', 'Lung Lesion', 'Pneumonia'],
                        'consolidation':['Pneumonia', 'Consolidation'],
                        'fracture':['Fracture'],
                        'pleural effusion':['Pleural Effusion'],
                        'pneumothorax':['Pneumothorax'],
                        'atelectasis':['Atelectasis']  }

primary_labels = {}

primary_labels['consolidation'] = ['Consolidation']
primary_labels['lung opacity'] = ['Lung Opacity']

df = pd.read_csv('nih_chexpert_labeled.csv')
df= df.fillna(-2)
# Initialize lists to store rows for the final table
rows = []
# Process each entry in the JSON data
for index, entry in df.iterrows():
    # subject_id = str(int(entry['subject_id']))
    # subject_id = ''

    # study_id = str(int(entry['study_id']))
    # study_id = entry['image_2'].rstrip('.jpg')
    study_id = entry['image_1']

    joined_labels_row = {'subjectid_studyid': f"{study_id}", 'type_annotation': 'labels'}

    labels_row = {'subjectid_studyid': f"{study_id}", 'type_annotation': 'labels'}
    for label_in_mimic in df.columns[2:]:
        labels_row[label_in_mimic] = entry[label_in_mimic]
    for label_to_include in columns_to_use:
        if label_to_include in columns_to_join_preds:
            for column_to_join in columns_to_join_preds[label_to_include]:
                if column_to_join in labels_row:
                    if label_to_include not in joined_labels_row:
                        joined_labels_row[label_to_include] = labels_row[column_to_join]
                    else:
                        if labels_row[column_to_join]==-1:
                            labels_row[column_to_join]=0.7
                        if joined_labels_row[label_to_include]==-1:
                            joined_labels_row[label_to_include]=0.7
                        if labels_row[column_to_join]==-2:
                            labels_row[column_to_join]=0.2
                        if joined_labels_row[label_to_include]==-2:
                            joined_labels_row[label_to_include]=0.2
                        if label_to_include in primary_labels and column_to_join in primary_labels[label_to_include]:
                            if labels_row[column_to_join]==0:
                                labels_row[column_to_join]=0.5
                            if joined_labels_row[label_to_include]==0:
                                joined_labels_row[label_to_include]=0.5
                        joined_labels_row[label_to_include] = max(joined_labels_row[label_to_include] , labels_row[column_to_join])
                        if joined_labels_row[label_to_include]==0.7:
                            joined_labels_row[label_to_include]=-1
                        if labels_row[column_to_join]==0.7:
                            labels_row[column_to_join]=-1
                        if labels_row[column_to_join]==0.2:
                            labels_row[column_to_join]=-2
                        if joined_labels_row[label_to_include]==0.2:
                            joined_labels_row[label_to_include]=-2
                        if labels_row[column_to_join]==0.5:
                                labels_row[column_to_join]=0
                        if joined_labels_row[label_to_include]==0.5:
                            joined_labels_row[label_to_include]=0
        else:
            if label_to_include in labels_row:
                joined_labels_row[label_to_include] = labels_row[label_to_include]
    rows.append(joined_labels_row)

# Create a DataFrame from the processed rows
df = pd.DataFrame(rows)

# Reset column names for better representation
# df.columns = ['study_id_subject_id', 'type_annotation'] + [key for key in abnormalities]
df.to_csv('./chexpert_nih_dataset_converted.csv')
# Display the resulting DataFrame