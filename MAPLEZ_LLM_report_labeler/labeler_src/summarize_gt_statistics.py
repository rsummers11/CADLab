# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# Ricardo Bigolin Lanfredi - 2023-07-12
# The input files are "experiment_test_annotations/ct_human_annotations.csv", 
# "experiment_test_annotations/mri_human_annotations.csv",
# "experiment_test_annotations/pet_human_annotations.csv", 
# "experiment_test_annotations/nih_human_annotations.csv" and 
# "experiment_test_annotations/mimic_human_annotations.csv",
# containing the annotations for each of the new hand-annotated groundtruths, 
# annotated from CXR reports.
# Two Latex formatted tables are outputed to stdout.
# Description:
# File used to generate Tables 8 and 9 in "Enhancing Chest X-ray
# Datasets with Privacy-Preserving LLMs and Multi-Type Annotations:
# A Data-Driven Approach for Improved Classification", containing
# the statistics for the new ground truthes used in the paper evaluation

import pandas as pd
from sortedcollections import OrderedSet
import re
import numpy as np

results = {}
for type_annotation in ['labels', 'location']:
    if type_annotation not in results:
        results[type_annotation] = {}
    for modality in ['ct','mri','pet']:
        if modality not in results[type_annotation]:
            results[type_annotation][modality] = {}
        if modality=='ct':
            abnormalities =  ['lung lesion', 'liver lesion', 'kidney lesion', 'adrenal gland abnormality','pleural effusion']
        elif modality=='mri':
            abnormalities=  ['liver lesion', 'kidney lesion', 'adrenal gland abnormality']
        elif modality=='pet':
            abnormalities =  ['hypermetabolic abnormality in the thorax', 'hypermetabolic abnormality in the abdomen', 'hypermetabolic abnormality in the pelvis']

        if modality=='ct':
            annotations = pd.read_csv('experiment_test_annotations/ct_human_annotations.csv')
        elif modality=='mri':   
            annotations = pd.read_csv('experiment_test_annotations/mri_human_annotations.csv')
        elif modality=='pet':
            annotations = pd.read_csv('experiment_test_annotations/pet_human_annotations.csv')

        if type_annotation == 'location':
            if modality=='ct':
                list_of_location_labels_per_abnormality = {
                    'lung lesion': ['right', 'left', 'upper','lower','middle'],
                    'liver lesion':['right', 'left'],
                    'kidney lesion':['right', 'left'],
                    'adrenal gland abnormality':['right', 'left'],
                    'pleural effusion':['right', 'left'],
                }
            elif modality=='mri':   
                list_of_location_labels_per_abnormality = {
                    "liver lesion":['right', 'left'],
                    "kidney lesion":['right', 'left'],
                    "adrenal gland abnormality":['right', 'left'],
                }

            list_of_replacements_labels_location = {'leftward': ['left'],
                                                            'left-sided':['left'],
                                                            'left-side':['left'],
                                                            'right-sided':['right'],
                                                            'right-side':['right'],
                                                            'rightward':['right'],
                                                            'infrahilar':['lower'],
                                                            'suprahilar':['upper'],
                                                            'fissural':['fissure'],
                                                            'fissures':['fissure'],
                                            'lingula':['left', 'lower'],
                                            'lingular':['left', 'lower'],
                                            'bilateral': ['right', 'left'], 
                                            'central':['perihilar'],
                                            'medial':['perihilar'],
                                            'medially':['perihilar'],
                                            'costrophenic':['base'],
                                            'bilaterally': ['right', 'left'], 
                                            'lungs': ['right', 'left'], 
                                            'biapical': ['right', 'left', 'apical'], 
                                            'apices': ['right', 'left', 'apical'], 
                                            'apexes': ['right', 'left', 'apical'], 
                                            'apex': ['apical'], 
                                            'apically': ['apical'], 
                                            'retrocardiac': ['left','lower', 'retrocardiac'],
                                            'mid': ['middle'], 
                                            'basilar':['base'], 
                                            'bases': ['base', 'right', 'left'], 
                                            'bibasilar': ['base', 'right', 'left'], 
                                            'chest walls': ['right', 'left', 'chest wall'], 
                                            'ventricular': ['ventricle'],
                                            'ventricles': ['right', 'left', 'ventricle'], 
                                            'atriums': ['right', 'left', 'atrium'], 
                                            'superior vena cava': ['svc'], 
                                            'below the diaphragms': ['below the diaphragm'], 
                                            'above the carinal':['above the carina'],
                                            'posterolateral': ['posterior', 'lateral'], 
                                            'apicolateral': ['apical', 'lateral'], 
                                            'ribs': ['rib'], 
                                            '3rd': ['third'], 
                                            '3': ['third'], 
                                            '4th': ['fourth'], 
                                            '4': ['fourth'], 
                                            'four': ['fourth'], 
                                            '5th': ['fifth'], 
                                            'five': ['fifth'], 
                                            '5': ['fifth'], 
                                            '6th': ['sixth'], 
                                            '6': ['sixth'], 
                                            'six': ['sixth'], 
                                            '7th': ['seventh'],
                                            '7': ['seventh'],  
                                            'seven': ['seventh'], 
                                            '8th': ['eighth'], 
                                            '8': ['eighth'],  
                                            'eight': ['eighth'], 
                                            '9th': ['ninth'], 
                                            '9': ['ninth'], 
                                            'nine': ['ninth'], 
                                            'clavicle': ['clavicular'], 
                                            'clavicles': ['clavicular', 'left', 'right'], 
                                            'vertebrae':['spine'], 
                                            'vertebra':['spine'], 
                                            'vertebral':['spine'],
                                            'spinal':['spine'],
                                            'ij':['jugular'],
                                            'caval atrial': ['cavoatrial'],
                                            'et': ['endotracheal'],
                                            'picc': ['pic'],
                                            'ng':['nasogastric']
                                            }

            str_labels_location = list(OrderedSet([item for sublist in list_of_location_labels_per_abnormality.values() for item in sublist]))
            allowed_locations = str_labels_location
            def get_location_labels_from_list_of_str(location):
                location_labels = np.zeros([len(allowed_locations)], dtype=np.int8)
                if location!='-1':
                    for location_label_str in set(str_labels_location + list(list_of_replacements_labels_location.keys())):                    
                        regexp = re.compile(fr'\b{location_label_str}\b', flags=re.IGNORECASE)
                        if regexp.search(location):
                            if location_label_str in list_of_replacements_labels_location.keys():
                                words_found = list_of_replacements_labels_location[location_label_str]
                            else:
                                words_found = [location_label_str]
                            for replacement_word in words_found:
                                if replacement_word in allowed_locations:
                                    location_labels[allowed_locations.index(replacement_word)] = 1
                return location_labels

        
        for abnormality in abnormalities:
            full_annotations = np.array([])
            if abnormality not in results[type_annotation][modality]:
                results[type_annotation][modality][abnormality] = {}
            table = []
            for row_index, row_annotations in annotations.iterrows():
                if row_annotations[abnormality]==row_annotations[abnormality]:
                    all_annotations = str(row_annotations[abnormality]).split(';')
                else:
                    all_annotations = []
                if type_annotation=='labels':
                    if '-3' in all_annotations:
                        annotation = np.array([-3])
                    elif '0' in all_annotations:
                        annotation = np.array([0])
                    elif '-1' in all_annotations:
                        annotation = np.array([-1])
                    elif len(all_annotations)==0:
                        annotation = np.array([-2])
                    else:
                        annotation = np.array([1])
                if type_annotation=='location':
                    if modality=='pet':
                        annotation = np.array([])
                    else:
                        location_annotation = [annotation for annotation in all_annotations if len(annotation)>1 and annotation not in ['mild','moderate','severe']]
                        annotation = get_location_labels_from_list_of_str(';'.join(location_annotation))
                        annotation = np.array([annotation[allowed_locations.index(allowed_location)] for allowed_location in allowed_locations if allowed_location in list_of_location_labels_per_abnormality[abnormality]])
                        annotation = np.stack(annotation).flatten()
                full_annotations = np.concatenate([full_annotations, annotation], axis = 0)
            results[type_annotation][modality][abnormality] = full_annotations
                # table.append({'annot':annotation,'pred':prediction})


for dataset in ['nih','mimic']:
    if dataset == 'nih':
        groundtruth = './experiment_test_annotations/nih_human_annotations.csv'
    else:
        groundtruth = './experiment_test_annotations/mimic_human_annotations.csv'
    abnormalities = ['enlarged cardiomediastinum',
    'cardiomegaly',
    'atelectasis',
    'consolidation',
    'lung edema',
    'fracture',
    'lung lesion',
    'pleural effusion',
    'pneumothorax',
    'support device',
    'lung opacity',
    'pleural other' ]

    annot = pd.read_csv(groundtruth)
    annot = annot[annot['type_annotation']=='labels']
    annot= annot.fillna('-2')
    for abnormality in abnormalities:
        if 'labels' not in results:
            results['labels'] = {}
        if dataset not in results['labels']:
            results['labels'][dataset] = {}
        if abnormality not in results['labels'][dataset]:
            results['labels'][dataset][abnormality] = {}
        results['labels'][dataset][abnormality] = pd.to_numeric(annot[abnormality]).values

    list_of_location_labels_per_abnormality = {
        'enlarged cardiomediastinum':['right', 'left', 'upper', 'lower', 'base', 'ventricle', 'atrium'],
        'cardiomegaly':['right', 'left', 'upper', 'lower', 'base', 'ventricle', 'atrium'],
        'lung lesion':['apical', 'middle', 'right', 'left', 'upper', 'lower', 'base', 'lateral', 'perihilar', 'retrocardiac'],
        'lung opacity':['apical', 'middle', 'right', 'left', 'upper', 'lower', 'base', 'lateral', 'perihilar', 'retrocardiac'],
        'lung edema':['apical', 'middle', 'right', 'left', 'upper', 'lower', 'base', 'lateral', 'perihilar', 'retrocardiac'],
        'consolidation':['apical', 'middle', 'right', 'left', 'upper', 'lower', 'base', 'lateral', 'perihilar', 'retrocardiac'],
        'atelectasis':['apical', 'middle', 'right', 'left', 'upper', 'lower', 'base', 'lateral', 'perihilar', 'retrocardiac'],
        'pneumothorax':['apical', 'middle', 'right', 'left', 'upper', 'lower', 'base', 'lateral', 'perihilar', 'retrocardiac'],
        'pleural effusion':['apical', 'middle', 'right', 'left', 'upper', 'lower', 'base', 'lateral', 'perihilar', 'retrocardiac', 'fissure'],
        'pleural other':['apical', 'middle', 'right', 'left', 'upper', 'lower', 'base', 'lateral', 'perihilar', 'retrocardiac', 'fissure'],
        'fracture':['middle', 'right', 'left', 'upper', 'lower', 'lateral', 'posterior', 'anterior', 'rib', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'clavicular', 'spine'],
        'support device':['apical', 'middle', 'right', 'left', 'upper', 'lower', 'base', 'pleural', 'chest wall', 'ventricle', 'atrium', 'svc', 'below the diaphragm', 'jugular', 'above the carina', 'cavoatrial', 'stomach','endotracheal','pic','subclavian', 'nasogastric', 'enteric', 'duodenum']
    }

    str_labels_location = list(OrderedSet([item for sublist in list_of_location_labels_per_abnormality.values() for item in sublist]))
    allowed_locations = str_labels_location

    def get_location_labels_from_list_of_str(label, location):
        location_labels = np.zeros([len(allowed_locations)], dtype=np.int8)
        if location!='-1':
            for location_label_str in set(str_labels_location + list(list_of_replacements_labels_location.keys())):                    
                regexp = re.compile(fr'\b{location_label_str}\b', flags=re.IGNORECASE)
                if regexp.search(location):
                    if location_label_str in list_of_replacements_labels_location.keys():
                        words_found = list_of_replacements_labels_location[location_label_str]
                    else:
                        words_found = [location_label_str]
                    for replacement_word in words_found:
                        if replacement_word in allowed_locations:
                            location_labels[allowed_locations.index(replacement_word)] = 1
        return location_labels

    def get_location_labels_per_row(row,columns_to_join_preds):
        for column_name in abnormalities:
            all_locations_collected = []
            if row[column_name]!='-1' and row[column_name]!=-1:
                all_locations_collected+= row[column_name].split(';')
            if len(all_locations_collected)==0:
                all_locations_collected = '-1'
            else:
                all_locations_collected = ';'.join(all_locations_collected)
            annotation = get_location_labels_from_list_of_str(column_name, all_locations_collected)
            annotation = np.array([annotation[allowed_locations.index(allowed_location)] for allowed_location in allowed_locations if allowed_location in list_of_location_labels_per_abnormality[column_name]])
            
            row[column_name] = annotation
        return row

    annot = pd.read_csv(groundtruth)
    annot = annot[annot['type_annotation']=='location']
    annot= annot.fillna('-1')
    annot = annot.apply(lambda row: get_location_labels_per_row(row,abnormalities), axis=1)
    for abnormality in abnormalities:
        if 'location' not in results:
            results['location'] = {}
        if dataset not in results['location']:
            results['location'][dataset] = {}
        if abnormality not in results['location'][dataset]:
            results['location'][dataset][abnormality] = {}
        results['location'][dataset][abnormality] = np.stack(annot[abnormality].values).flatten()


    for abnormality in abnormalities:
        annot = pd.read_csv(groundtruth)
        annot = annot[annot['type_annotation']=='severity']
        annot= annot.fillna('-1')
        if 'severity' not in results:
            results['severity'] = {}
        if dataset not in results['severity']:
            results['severity'][dataset] = {}
        if abnormality not in results['severity'][dataset]:
            results['severity'][dataset][abnormality] = {}
        results['severity'][dataset][abnormality] = pd.to_numeric(annot[abnormality]).values


# table formatting
labels_dict = {'stable':-3,'not mentioned':-2,'uncertain':-1,'absent':0,'present':1}
severities_dict = {'mild':1,'moderate':2,'severe':3}
str_labels_mimic = {
        'atelectasis':'Atelectasis',
    'cardiomegaly':'Cardiomegaly',
    'consolidation':'Consolidation',
    'enlarged cardiomediastinum':'Enlarged cardiomediastinum',
    'fracture':'Fracture',
    'lung edema':'Edema',
    'lung lesion':'Lung lesion',
    'lung opacity':'Lung opacity',
    'pleural effusion':'Pleural effusion',
    'pleural other':'Pleural other' ,
    'pneumothorax':'Pneumothorax',
    'support device':'Support device',
'liver lesion':'Liver lesion', 
'kidney lesion':'Kidney lesion', 
'adrenal gland abnormality':'Adrenal gland abnormality',
'hypermetabolic abnormality in the thorax':'Hypermet. thorax', 
'hypermetabolic abnormality in the abdomen':'Hypermet. abdomen', 
'hypermetabolic abnormality in the pelvis':'Hypermet. pelvis'}

datasets = {'nih':'NIH','mimic':'MIMIC','ct':'CT','mri':'MRI','pet':'PET'}

row_names_list = [{'gt_present': '``Present"',
'gt_absent': '``Absent"',
'gt_uncertain': '``Uncertain"',
'gt_not mentioned': 'NM',
'gt_stable': '``Stable"',
'n':'\\#',
},
 {
'severities_present': 'Sevs.',
'severities_mild': 'Mild',
'severities_moderate': 'Moderate',
'severities_severe': 'Severe',
'location_positive': 'Loc.+',
'location_total': 'Loc.\\#',
}]

table = []

for abnormality in str_labels_mimic:
    row = {}
    for dataset in datasets:
        if abnormality in results['labels'][dataset]:
            row['abnormality'] = abnormality
            this_df = results['labels'][dataset][abnormality]
            row[f'{dataset}_n'] = len(this_df)
            for category in labels_dict:
                if category in ['stable','not mentioned','uncertain'] and dataset=='pet':
                    continue
                row[f'{dataset}_gt_'+category] = ((this_df==labels_dict[category])*1).sum()
            if dataset in results['severity']:

                this_df = results['severity'][dataset][abnormality]
                
                row[f'{dataset}_severities_present'] = ((this_df>0)*1).sum()
                for category in severities_dict:
                    row[f'{dataset}_severities_'+category] =  ((this_df==severities_dict[category])*1).sum()
            if dataset in results['location']:
                this_df = results['location'][dataset][abnormality]
                row[f'{dataset}_location_positive'] = ((this_df==1)*1).sum()
                row[f'{dataset}_location_total'] = len(this_df)
    table.append(row)
summarized_statistics = pd.DataFrame(table)

precision = 0
precision_percentage = 2
table_string = ''

def format_with_two_significant_digits(number):
    # Check if the number is a whole number and has less than two significant digits
    if number <10:
        # Add one decimal place to make it two significant digits
        return f"{number:.1f}"
    else:
        # Else, format the number to have two significant digits
        return f"{number:.2g}"

for row_names in row_names_list:
    table_string+= 'Abnormality'
    table_string += ' & '
    table_string+= 'Data'
    table_string += ' & '

    for row_name in row_names:
        table_string+= row_names[row_name]
        table_string += ' & '

    table_string+='\\\\\n'
    table_string+='\\midrule'
    table_string+='\n'


    for abnormality in str_labels_mimic:
        for dataset in datasets:
            row_string = ''
            for row_name in row_names:
                if f'{dataset}_{row_name}' in summarized_statistics:
                    value = summarized_statistics[summarized_statistics['abnormality']==abnormality][f'{dataset}_{row_name}'].values[0]
                    if 'location' in row_name:
                        total = summarized_statistics[summarized_statistics['abnormality']==abnormality][f'{dataset}_location_total'].values[0]
                    else:
                        total = summarized_statistics[summarized_statistics['abnormality']==abnormality][f'{dataset}_n'].values[0]
                    if value==value:
                        percentage = value/total*100
                        if 'n'!=row_name and 'total' not in row_name:
                            string_to_add = f'{value:.{precision}f} ({format_with_two_significant_digits(percentage)}\\%)'
                        else:
                            string_to_add = f'{value:.{precision}f}'
                    else:
                        string_to_add = '-'
                else:
                    string_to_add = '-'
                row_string+= string_to_add
                row_string += ' & '
            if len(row_string.replace('&','').replace('-','').replace(' ',''))>0:
                table_string+=str_labels_mimic[abnormality]
                table_string += ' & '
                table_string+=datasets[dataset]
                table_string += ' & '
                table_string+=row_string
                table_string+='\\\\\n'

table_string = table_string.replace('& \\\\\n', '\\\\\n')
table_string = table_string.replace('.0 ', ' ')
print(table_string)



