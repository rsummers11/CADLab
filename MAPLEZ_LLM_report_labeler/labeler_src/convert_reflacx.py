# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# Ricardo Bigolin Lanfredi - 2023-14-12
# The script takes the ./reflacx/main_data/metadata_phase_1.csv, ./reflacx/main_data/metadata_phase_2.csv, ./reflacx/main_data/metadata_phase_3.csv
# and ./mimic/mimic-cxr-2.0.0-metadata.csv
# files as input and outputs to reflacx_phase2_dataset_converted.csv
# Description:
# Script to convert the REFLACX dataset labels to the format used by our labeler

import json
import pandas as pd

def get_probability_from_number(probability_number):
    probability_number = int(probability_number)
    if probability_number==5:
        return 90
    if probability_number==4:
        return 75
    if probability_number==3:
        return 50
    if probability_number==2:
        return 25
    if probability_number==1:
        return 10
    if probability_number==0:
        return 0

columns_to_use = ['pneumothorax','cardiomegaly','lung opacity','consolidation','lung edema', 'atelectasis', 'pleural effusion', 'fracture']

columns_to_join_preds = {'consolidation':['Consolidation'],
                         'cardiomegaly':['Enlarged cardiac silhouette'],
                        'lung opacity':['Interstitial lung disease', 'Pulmonary edema','Consolidation', 'Atelectasis','Enlarged hilum', 'Groundglass opacity', 'Lung nodule or mass'],
                        'lung edema':['Pulmonary edema'],
                        'pneumothorax':['Pneumothorax'],
                        'atelectasis':['Atelectasis'],
                        'pleural effusion':['Pleural abnormality'],
                        'fracture':['Fracture'],
                         }

primary_labels = {}

main_data_folder = './reflacx/main_data/'

def main(phase):
    if phase==1:
        df = pd.read_csv(main_data_folder + 'metadata_phase_1.csv')
    if phase==2:
        df = pd.read_csv(main_data_folder + 'metadata_phase_2.csv')
    if phase==3:
        df = pd.read_csv(main_data_folder + 'metadata_phase_3.csv')
    df= df.fillna(0)
    # Initialize lists to store rows for the final table
    rows = []

    metadata_chexpert = pd.read_csv('./mimic/mimic-cxr-2.0.0-metadata.csv')

    # Process each entry in the JSON data
    for index, entry in df.iterrows():
        subject_id = str(int(entry['subject_id']))
        study_id = metadata_chexpert[metadata_chexpert['dicom_id'] == entry['dicom_id']]['study_id'].values[0]

        joined_labels_row = {'subjectid_studyid': f"{subject_id}_{study_id}", 'type_annotation': 'labels'}

        labels_row = {'subjectid_studyid': f"{subject_id}_{study_id}", 'type_annotation': 'labels'}

        joined_probability_row = {'subjectid_studyid': f"{subject_id}_{study_id}", 'type_annotation': 'probability'}

        probability_row = {'subjectid_studyid': f"{subject_id}_{study_id}", 'type_annotation': 'probability'}

        for label_in_mimic in list(df.columns[8:19]) + list(df.columns[20:23]):
            labels_row[label_in_mimic] = 1 if int(entry[label_in_mimic])==5 else -1 if int(entry[label_in_mimic])>0 else 0
            probability_row[label_in_mimic] = get_probability_from_number(entry[label_in_mimic])
        for label_to_include in columns_to_use:
            if label_to_include in columns_to_join_preds:
                for column_to_join in columns_to_join_preds[label_to_include]:
                    if column_to_join in probability_row:
                        if label_to_include not in joined_probability_row:
                            joined_probability_row[label_to_include] = probability_row[column_to_join]
                        else:
                            joined_probability_row[label_to_include] = max(joined_probability_row[label_to_include] , probability_row[column_to_join])
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
                if label_to_include in probability_row:
                    joined_probability_row[label_to_include] = probability_row[label_to_include]
                if label_to_include in labels_row:
                    joined_labels_row[label_to_include] = labels_row[label_to_include]
        rows.append(joined_labels_row)
        rows.append(joined_probability_row)

    # Create a DataFrame from the processed rows
    df = pd.DataFrame(rows)

    if phase==1:
        df.to_csv('./reflacx_phase1_dataset_converted.csv')
    if phase==2:
        df.to_csv('./reflacx_phase2_dataset_converted.csv')
    if phase==3:
        df.to_csv('./reflacx_dataset_converted.csv')

if __name__=='__main__':
    main(1)
    main(2)
    main(3)
    df1 = pd.read_csv('./reflacx_phase1_dataset_converted.csv')
    df2 = pd.read_csv('./reflacx_phase2_dataset_converted.csv')
    concatenated_df = pd.concat([df1, df2], ignore_index=True)
    concatenated_df.to_csv('./reflacx_phase12_dataset_converted.csv', index=False)