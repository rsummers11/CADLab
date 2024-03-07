# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# Ricardo Bigolin Lanfredi - 2023-14-12
# This script used the following files as inputs:
# ./MIMIC-CXR/mimic-cxr-jpg-2.0.0.physionet.org/mimic-cxr-2.0.0-chexpert.csv.gz
# ./MIMIC-CXR/mimic-cxr-jpg-2.0.0.physionet.org/mimic-cxr-2.0.0-split.csv.gz
# ./MIMIC-CXR/mimic-cxr-jpg-2.0.0.physionet.org/mimic-cxr-2.0.0-metadata.csv.gz
# The output is train_df_all.csv val_df_all.csv test_df_all.csv
# Description:
# Script used to pregenerate csv files containing information for our splits of the mimic-cxr dataset.
# This script should be run before using the mimic-cxr pytorch dataset class

import pandas as pd
import os
from global_paths import jpg_path, mimic_dir

label_csv = os.path.join(mimic_dir + 'mimic-cxr-2.0.0-chexpert.csv.gz')
label_df = pd.read_csv(label_csv, compression='gzip')

split_csv = os.path.join(mimic_dir + 'mimic-cxr-2.0.0-split.csv.gz')
split_df = pd.read_csv(split_csv, compression='gzip')

metadata_csv = os.path.join(mimic_dir + 'mimic-cxr-2.0.0-metadata.csv.gz')
metadata_df = pd.read_csv(metadata_csv, compression='gzip')

# Merge split_df and label_df
# to get a single df with labels, split and image_name for each image
images_df = pd.merge(left=label_df, right=split_df, left_on = ['subject_id', 'study_id'], right_on=['subject_id', 'study_id'], how='inner')
images_df = pd.merge(left=metadata_df, right=images_df, left_on = ['subject_id', 'study_id','dicom_id'], right_on=['subject_id', 'study_id','dicom_id'], how='inner')
print(len(images_df))
images_df['ViewPosition'] = images_df['ViewPosition'].fillna('')
images_df = images_df[images_df['ViewPosition'].str.contains(r'AP|PA')]
print(len(images_df))
# images_df = images_df[images_df['ViewPosition']=='PA']

# images_df = images_df[images_df['PatientOrientationCodeSequence_CodeMeaning']=='Erect']

def getImgList(image_path, jpg_path):
    image_list_jpg = []
    with open(image_path) as f:
        image_list = f.readlines()
    for path in image_list:
        temp_path = jpg_path + '/files/' + path.split('files')[-1]
        temp_path = temp_path.replace('.dcm', '.jpg')
        image_list_jpg.append(temp_path.strip())
    return image_list_jpg

final_df = images_df

# Rearanging columns
cols = final_df.columns.to_list()
cols = cols[:2] + cols[-3:] + cols[2:-3]
final_df = final_df[cols]
final_df.head()

# Split final_df into Train, Validation and Test dfs
# using the val_mimicid.txt file to load subject_ids that should be moved from training set to validation set
train_df = final_df.loc[final_df['split'].isin(['train'])]
train_df = train_df.drop('split', axis=1)
val_df = final_df.loc[final_df['split'].isin(['validate'])]
val_df = val_df.drop('split', axis=1)
test_df = final_df[final_df['split'] == 'test']
test_df = test_df.drop('split', axis=1)

train_df.to_csv('train_df_all.csv', index=False)
val_df.to_csv('val_df_all.csv', index=False)
test_df.to_csv('test_df_all.csv', index=False)

