# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# Ricardo Bigolin Lanfredi - 2023-07-12
# Use `python <python_file> --help` to check inputs, outputs and their meaning. 
# The format of the output follows the rules defined in the README file
# Description:
# File used to convert the acquired outputs from the Vicuna paper 
# (Feasibility of Using the Privacy-preserving Large Language Model
# Vicuna for Labeling Radiology Reports, Mukherjee et al., 
# 10.1148/radiol.231147) to the format required by the calculate_auc_v2.py
# file

import json
import pandas as pd
import argparse

columns_to_use = ['pneumothorax','pleural effusion','fracture', 'cardiomegaly','lung opacity','consolidation','lung edema','atelectasis']

columns_to_join_preds = {'cardiomegaly':['Cardiomegaly_predicted'],
                            'lung edema':['Edema_predicted'],
                            'lung opacity':['Lung Opacity_predicted', 'Consolidation_predicted', 'Edema_predicted', 'Atelectasis_predicted', 'Lung Lesion_predicted', 'Pneumonia_predicted'],
                            'consolidation':['Pneumonia_predicted', 'Consolidation_predicted'],
                            'fracture':['Fracture_predicted'],
                            'pleural effusion':['Pleural Effusion_predicted'],
                            'pneumothorax':['Pneumothorax_predicted'],
                            'atelectasis':['Atelectasis_predicted']   }

primary_labels = {}

primary_labels['consolidation'] = ['Consolidation_predicted']
primary_labels['lung opacity'] = ['Lung Opacity_predicted']

def main(args):
    df = pd.read_csv(args.input_vicuna_file)
    df= df.fillna(-2)
    df['image_2_stripped'] = df['study_id'].str.rstrip('.jpg')
    nih_correspondance = pd.read_csv('image_1_image_2_nih_conversion.csv')
    nih_correspondance['image_2_stripped'] = nih_correspondance['image_2'].str.rstrip('.jpg')
    df = pd.merge(df, nih_correspondance[['image_1','image_2_stripped']], on='image_2_stripped', how='inner')
    print(len(df))
    df['study_id'] = df['image_1']

    # Initialize lists to store rows for the final table
    rows = []
    # Process each entry in the JSON data
    for index, entry in df.iterrows():

        # study_id = str(int(entry['study_id']))
        study_id = entry['study_id']


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

    df.to_csv(args.output_file, index = False)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_vicuna_file", type=str, default="./vicuna_results_nih.csv", help='''
    Filepath to the file containing the unconverted Vicuna outputs for the test set of the NIH ChestXray14 dataset.''')
    parser.add_argument("--output_file", type=str, default='./vicuna_nih_dataset_converted.csv', help='''
        Filepath where the converted outputs should be saved
    ''')
    args = parser.parse_args()
    main(args)