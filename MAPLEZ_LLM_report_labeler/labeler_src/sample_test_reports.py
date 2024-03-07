# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# Ricardo Bigolin Lanfredi - 2023-08-12
# This script takes these files as input: 
# - ./mimic/mimic-cxr-2.0.0-split.csv.gz (from the MIMIC-CXR dataset)
# - mimic_reports.csv (a csv containing all of the MIMIC-CXR reports)
# - nih_reports.csv (a csv containing all of the NIH ChestXray14 test reports)
# - human_annotation.csv (first validation set for the labeler, not included with the code)
# The outputs are written to './nih_reports_to_annotate_by_human.csv' and './mimic_reports_to_annotate_by_human.csv'
# Description:
# This script generates the list of reports that should be 
# manually labeled to use as groundtruth for evaluating the labelers

import pandas as pd
import os

mimic_dir = './mimic/' # Location of the tables mimic-cxr-2.0.0-chexpert.csv and mimic-cxr-2.0.0-split.csv from the MIMIC-CXR-JPG dataset

# open all test reports for mimic
split_csv = os.path.join(mimic_dir + 'mimic-cxr-2.0.0-split.csv.gz')
split_df = pd.read_csv(split_csv, compression='gzip')
print(split_df.columns)
split_df['subjectid_studyid'] = split_df['subject_id'].astype(str) + '_' + split_df['study_id'].astype(str)

reports_df = pd.read_csv('mimic_reports.csv')
images_df = pd.merge(left=split_df, right=reports_df, left_on = ['subjectid_studyid'], right_on=['subjectid_studyid'], how='inner')
images_df = images_df[images_df['split']=='test']
# open all test reports for nih
nih_test_reports = pd.read_csv('nih_reports.csv')
nih_test_reports['subjectid_studyid'] = nih_test_reports['subjectid_studyid'].str[1:]

# Remove the reports from validation set
to_exclude = pd.read_csv('human_annotation.csv')
to_exclude['image2'] = to_exclude['image2'].str[:-4]
nih_test_reports = nih_test_reports[~nih_test_reports['subjectid_studyid'].isin(to_exclude['image2'].values)]

# sample 202 from one, 405 from another
sampled_nih = nih_test_reports.sample(n=202, random_state=42)
sampled_mimic = images_df.sample(n=355, random_state=42)
sampled_nih = sampled_nih[['subjectid_studyid','report']]
sampled_mimic = sampled_mimic[['subjectid_studyid','report']]

pd.concat([sampled_nih] * 4, ignore_index=True).to_csv('./nih_reports_to_annotate_by_human.csv')
pd.concat([sampled_mimic] * 4, ignore_index=True).to_csv('./mimic_reports_to_annotate_by_human.csv')