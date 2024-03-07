# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# Ricardo Bigolin Lanfredi - 2024-20-02
# Description:
# file called from create_raw_table.py

import pandas as pd
from sortedcollections import OrderedSet
import re
import numpy as np
import os

def main(type_annotation, destination_folder, modality, prediction_file, annotation_file):
    # type_annotation = 'location'
    # destination_folder = '~/projects/llm'
    # modality = 'mri'

    if modality=='ct':
        abnormalities =  ['lung lesion', 'liver lesion', 'kidney lesion', 'adrenal gland abnormality','pleural effusion']
    elif modality=='mri':
        abnormalities=  ['liver lesion', 'kidney lesion', 'adrenal gland abnormality']
    elif modality=='pet':
        abnormalities =  ['hypermetabolic abnormality in the thorax', 'hypermetabolic abnormality in the abdomen', 'hypermetabolic abnormality in the pelvis']

    predictions = None
    annotations = None

    if type_annotation == 'labels':
        word_type = 'labeling'
    if type_annotation == 'location':
        word_type = 'loc'

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
        

    to_return = {}
    for abnormality in abnormalities:
        if os.path.exists(f'{destination_folder}/{abnormality}_llm_{modality}_{word_type}_outputs.csv'):
            file_with_annotations = pd.read_csv(f'{destination_folder}/{abnormality}_llm_{modality}_{word_type}_outputs.csv')
            full_annotations = file_with_annotations['annot'].values
            full_predictions = file_with_annotations['pred'].values
            to_return[abnormality] = full_annotations,full_predictions
            continue

        if annotations is None:
            annotations = pd.read_csv(annotation_file)
        if predictions is None:
            predictions = pd.read_csv(prediction_file)
            predictions = predictions[predictions['type_annotation']==type_annotation]
            predictions = pd.merge(annotations['subjectid_studyid'], predictions, on = 'subjectid_studyid', how = 'inner').reset_index()
        full_annotations = np.array([])
        full_predictions = np.array([])
        table = []
        for row_index, row_annotations in annotations.iterrows():
            row_predictions = predictions.iloc[row_index]
            if row_annotations[abnormality]==row_annotations[abnormality]:
                all_annotations = str(row_annotations[abnormality]).split(';')
            else:
                all_annotations = []
            if type_annotation=='labels':
                prediction = np.array([(row_predictions[abnormality] in [-1,1,'-1','1'])*1])
                annotation = np.array([(len(all_annotations)>0 and '-3' not in all_annotations and '0' not in all_annotations)*1])
            if type_annotation=='location':
                if row_predictions[abnormality]!='-1' and row_predictions[abnormality]!=-1:
                    prediction = get_location_labels_from_list_of_str(row_predictions[abnormality])
                else:
                    prediction = get_location_labels_from_list_of_str('')
                location_annotation = [annotation for annotation in all_annotations if len(annotation)>1 and annotation not in ['mild','moderate','severe']]
                annotation = get_location_labels_from_list_of_str(';'.join(location_annotation))
                prediction =  np.stack(prediction).flatten()
                annotation = np.stack(annotation).flatten()
            full_annotations = np.concatenate([full_annotations, annotation], axis = 0)
            full_predictions = np.concatenate([full_predictions, prediction], axis = 0)
            # table.append({'annot':annotation,'pred':prediction})
        pd.DataFrame({'annot':full_annotations,'pred':full_predictions}).to_csv(f'{destination_folder}/{abnormality}_llm_{modality}_{word_type}_outputs.csv')
        to_return[abnormality] = full_annotations,full_predictions
    return to_return
