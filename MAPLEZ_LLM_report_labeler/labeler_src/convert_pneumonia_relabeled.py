# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# Ricardo Bigolin Lanfredi - 2023-14-12
# The script takes the pneumonia_dataset_mapping.csv,
# stage_2_train_labels.csv and image_1_image_2_nih_conversion.csv
# file as input and outputs to pneumonia_relabeled_dataset_converted.csv
# Description:
# Script to convert the Pneumonia dataset labels to the format used by our labeler

import json
import pandas as pd

challenge_labels = pd.read_csv('stage_2_train_labels.csv')
challenge_correspondance = pd.read_csv('pneumonia_dataset_mapping.csv')
challenge_correspondance['patientId'] = challenge_correspondance['subset_img_id']
challenge_correspondance['image_1'] = challenge_correspondance['img_id'].str.rstrip('.png')

challenge_labels = pd.merge(challenge_labels, challenge_correspondance[['image_1','patientId']], on='patientId', how='inner')
challenge_labels['subjectid_studyid'] = './images/' + challenge_labels['image_1'] + '.png'
challenge_labels['consolidation'] = challenge_labels['Target']
challenge_labels['lung opacity'] = challenge_labels['Target']
challenge_labels['type_annotation'] = 'labels'
challenge_labels = challenge_labels[['subjectid_studyid','type_annotation','consolidation','lung opacity']]
challenge_labels = challenge_labels.drop_duplicates()
challenge_labels.to_csv('./pneumonia_relabeled_dataset_converted.csv')