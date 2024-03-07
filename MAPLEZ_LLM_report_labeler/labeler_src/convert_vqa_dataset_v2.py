# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# Ricardo Bigolin Lanfredi - 2023-14-12
# The script takes the vqa/cad_disease.json fiel as input and outputs to vqa_dataset_converted.csv
# Description:
# Script to convert the Medical Diff VQA dataset labels to the format used by our labeler

import json
import pandas as pd

columns_to_use = ['pneumothorax','pleural effusion','fracture', 'cardiomegaly','lung opacity','consolidation','lung edema','atelectasis']

columns_to_join_preds = {'cardiomegaly':['cardiomegaly', 'enlargement of the cardiac silhouette'],
                        'lung edema':['edema', 'vascular congestion', 'heart failure', 'hilar congestion'],
                        'lung opacity':['lung opacity', 'consolidation', 'edema', 'vascular congestion', 'atelectasis', 'heart failure', 'hilar congestion', 'pneumonia'],
                        'consolidation':['pneumonia', 'consolidation'] }

primary_labels = {}

primary_labels['consolidation'] = ['consolidation']
primary_labels['lung opacity'] = ['lung opacity']
primary_labels['lung edema'] = ['edema']
primary_labels['cardiomegaly'] = ['cardiomegaly']

def get_probability_from_word(probability_word):
    probability_word = probability_word.lower().strip()
    if probability_word in ['positive','change in']:
        return 100
    if probability_word in ['probable','probabl', 'likely', 'may', 'could', 'potential']:
        return 70
    if probability_word in ['might', 'possibl', 'possible']:
        return 50
    if probability_word in ['unlikely','not exclude', 'cannot be verified', 'difficult rule out', 'not ruled out', 'cannot be accurately assessed', 'not rule out', 'impossible exclude', 'cannot accurately assesses', 'cannot be assessed', 'cannot be identified', 'cannot be confirmed', 'cannot be evaluated', 'difficult exclude']:
        return 30
    if probability_word in ['no', 'without', 'negative', 'clear of', 'exclude', 'lack of', 'rule out', 'ruled out']:
        return 0
    print(probability_word)
    1/0

# Load the JSON data
with open('vqa/cad_disease.json', 'r') as json_file:
    data = json.load(json_file)

# Initialize lists to store rows for the final table
rows = []
# Process each entry in the JSON data
for entry in data:
    study_id = entry['study_id']
    subject_id = entry['subject_id']
    
    # Process labels
    labels = entry['entity']
    
    # Create a row for labels (1) and absences (0)
    probability_row = {'subjectid_studyid': f"{subject_id}_{study_id}", 'type_annotation': 'probability'}
    labels_row = {'subjectid_studyid': f"{subject_id}_{study_id}", 'type_annotation': 'labels'}
    locations_row = {'subjectid_studyid': f"{subject_id}_{study_id}", 'type_annotation': 'location'}
    severity_row = {'subjectid_studyid': f"{subject_id}_{study_id}", 'type_annotation': 'severity'}
    
    # Process locations associated with each type
    for label in labels:
        
        locations = (entry['entity'][label]['location'] or []) +  (entry['entity'][label]['location2'] or [])
        if len(locations)>0:
            locations_row[label] =  ';'.join(locations)        
        
        # Process severity words
        severity_words = (entry['entity'][label]['level'] or [])+ (entry['entity'][label]['level2'] or [])
        severities = [-1]
        for severity_word in severity_words:
            severities.append({'moderate':2,'mild':1,'minimal':1,'small':1,'massive':3,'severe':3,'moderate to severe':3,'mild to moderate':2,
            'subtle':1,'increasing':-1,'minimally':1, 'acute':-1, 'moderately':2,'mildly':1,'moderate to large':3,'decreasing':-1,'trace':1,'minor':1}[severity_word])
        severity_row[label] = max(severities)


        probability_score = get_probability_from_word(labels[label]['probability']) if 'probability' in labels[label] else 100
        probability_row[label] = probability_score

        if 'probability_score' not in labels[label] or labels[label]['probability_score']!=0:
            if ('probability_score' not in labels[label] or labels[label]['probability_score']==3):
                labels_row[label] = 1
            elif labels[label]['probability_score']==-3:
                labels_row[label] = 0
            else:
                labels_row[label] = -1
    
    joined_probability_row = {'subjectid_studyid': f"{subject_id}_{study_id}", 'type_annotation': 'probability'}
    joined_labels_row = {'subjectid_studyid': f"{subject_id}_{study_id}", 'type_annotation': 'labels'}
    joined_locations_row = {'subjectid_studyid': f"{subject_id}_{study_id}", 'type_annotation': 'location'}
    joined_severity_row = {'subjectid_studyid': f"{subject_id}_{study_id}", 'type_annotation': 'severity'}

    for label_to_include in columns_to_use:
        if label_to_include in columns_to_join_preds:
            for column_to_join in columns_to_join_preds[label_to_include]:
                if column_to_join in locations_row:
                    if label_to_include not in joined_locations_row:
                        joined_locations_row[label_to_include] = locations_row[column_to_join]
                    else:
                        joined_locations_row[label_to_include] = joined_locations_row[label_to_include].split(';') + locations_row[column_to_join].split(';')
                        res = []
                        [res.append(x) for x in joined_locations_row[label_to_include] if x not in res]
                        joined_locations_row[label_to_include] = ';'.join(res)
                if column_to_join in severity_row:
                    if label_to_include not in joined_severity_row:
                        joined_severity_row[label_to_include] = severity_row[column_to_join]
                    else:
                        joined_severity_row[label_to_include] = max(joined_severity_row[label_to_include] , severity_row[column_to_join])
                if column_to_join in probability_row:
                    if label_to_include not in joined_probability_row:
                        joined_probability_row[label_to_include] = probability_row[column_to_join]
                    else:
                        joined_probability_row[label_to_include] = max(joined_probability_row[label_to_include] , probability_row[column_to_join])
                if column_to_join not in labels_row:
                    labels_row[column_to_join] = -2
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
            if label_to_include in locations_row:
                joined_locations_row[label_to_include] = locations_row[label_to_include]
            if label_to_include in severity_row:
                joined_severity_row[label_to_include] = severity_row[label_to_include]
            if label_to_include in probability_row:
                joined_probability_row[label_to_include] = probability_row[label_to_include]
            if label_to_include in labels_row:
                joined_labels_row[label_to_include] = labels_row[label_to_include]
            else:
                joined_labels_row[label_to_include] = -2
    rows.append(joined_locations_row)
    rows.append(joined_severity_row)
    rows.append(joined_probability_row)
    rows.append(joined_labels_row)

# Create a DataFrame from the processed rows
df = pd.DataFrame(rows)

# Reset column names for better representation
# df.columns = ['study_id_subject_id', 'type_annotation'] + [key for key in abnormalities]
df.to_csv('./vqa_dataset_converted.csv')
# Display the resulting DataFrame