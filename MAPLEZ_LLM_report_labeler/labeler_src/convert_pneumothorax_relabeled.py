# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# Ricardo Bigolin Lanfredi - 2023-14-12
# The script takes the relabeled/test_manifest_pneumothorax_relabeled.json
# and image_1_image_2_nih_conversion.csv
# files as input and outputs to pneumothorax_relabeled_dataset_converted.csv
# Description:
# Script to convert the Pneumothorax dataset labels to the format used by our labeler


import json
import pandas as pd

with open('relabeled/test_manifest_pneumothorax_relabeled.json', 'r') as file:
    json_data = file.read()
# Split the data into individual JSON objects
json_objects = json_data.strip().split('}{')

rows = []

start = 0
for end, char in enumerate(json_data):
    if char == '}':
        json_obj = json_data[start:end+1]
        entry = json.loads(json_obj)
        # Process the data
        start = end + 1
        study_id = entry['image_filepath'].lstrip('/data/').rstrip('.jpg').rstrip('.png')
        
        # Process labels
        label = 1 if (entry['label'] == 'positive') else 0
        
        labels_row = {'image_1': f"{study_id}", 'type_annotation': 'labels', 'pneumothorax': label}
        
        rows.append(labels_row)

# Create a DataFrame from the processed rows
df = pd.DataFrame(rows)
df['subjectid_studyid'] ='./images/' + df['image_1'] + '.png'
# Reset column names for better representation
# df.columns = ['study_id_subject_id', 'type_annotation'] + [key for key in abnormalities]
df.to_csv('./pneumothorax_relabeled_dataset_converted.csv')
# Display the resulting DataFrame