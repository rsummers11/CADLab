# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# Ricardo Bigolin Lanfredi - 2023-14-12
# Description:
# Auxiliary file containing a pytorch dataset for the mimic-cxr dataset

import imageio
import numpy as np
from torch.utils.data import Dataset
from list_labels import str_labels_mimic as str_labels
from global_paths import jpg_path
import pandas as pd
from list_labels import str_labels_location, translation_mimic_to_new_labels
import re
from list_labels import str_labels_location as list_of_words
from list_labels import list_of_replacements_labels_location as list_of_replacements
from list_labels import list_of_uncertain_labels_location, list_of_negative_labels_location, list_of_location_labels_per_abnormality

from threading import Lock

s_print_lock = Lock()

def pre_process_path(dicom_path):
    temp_path = jpg_path + '/files/' + dicom_path.split('files')[-1]
    temp_path = temp_path.replace('.dcm', '.jpg')
    return temp_path.strip()

class MIMICCXRDataset(Dataset):
    def __init__(self, df, df_chexbert, df_new_labels_llm, df_new_labels_vqa, df_labels_reflacx):
        self.df = df
        self.dataset_size = len(self.df)
        
        self.df[str_labels] = (self.df[str_labels].fillna(-2))
        # self.locations_indices_search = pd.read_csv('clip_locations_embeddings.csv')
        self.df_new_labels_llm = df_new_labels_llm
        self.df_new_labels_vqa = df_new_labels_vqa
        self.df_labels_reflacx = df_labels_reflacx
        self.df_chexbert = df_chexbert
        self.df_chexbert[str_labels] = (self.df_chexbert[str_labels].fillna(-2))
    
    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx):
        print(idx)
        if idx >= len(self):
            raise StopIteration
        pid = 'p' + str(self.df.iloc[idx]["subject_id"])
        filepath = jpg_path + '/files/' + pid[:3] + '/' + pid + '/s' + str(self.df.iloc[idx]["study_id"]) + '/' + self.df.iloc[idx]["dicom_id"] + '.jpg'
        img = imageio.imread(pre_process_path(filepath))
        mimic_gt = np.zeros([len(str_labels)])
        chexbert_gt = np.zeros([len(str_labels)])
        new_gt = np.zeros([len(str_labels)])
        probabilities = np.zeros([len(str_labels)])
        location_labels = np.zeros([len(str_labels),len(str_labels_location)])-2
        severities = np.zeros([len(str_labels)])
        location_vector_index = np.zeros([len(str_labels)])-1
        unchanged_uncertainties = np.zeros([len(str_labels)])
        vqa_new_gt = np.zeros([len(str_labels)])
        vqa_probabilities = np.zeros([len(str_labels)])
        vqa_location_labels = np.zeros([len(str_labels),len(str_labels_location)])-2
        vqa_severities = np.zeros([len(str_labels)])
        vqa_location_vector_index = np.zeros([len(str_labels)])-1
        reflacx_new_gt = np.zeros([len(str_labels)])
        reflacx_probabilities = np.zeros([len(str_labels)])

        for dataset in ['llm','vqa', 'reflacx']:
            if dataset=='llm':
                rows_new_labels_this_case = self.df_new_labels_llm[self.df_new_labels_llm['subjectid_studyid'] ==  str(self.df.iloc[idx]["subject_id"]) + '_' + str(self.df.iloc[idx]["study_id"])]
            elif dataset=='vqa':
                rows_new_labels_this_case = self.df_new_labels_vqa[self.df_new_labels_vqa['subjectid_studyid'] ==  str(self.df.iloc[idx]["subject_id"]) + '_' + str(self.df.iloc[idx]["study_id"])]
            elif dataset=='reflacx':
                rows_new_labels_this_case = self.df_labels_reflacx[self.df_labels_reflacx['subjectid_studyid'].astype(str) ==  str(self.df.iloc[idx]["subject_id"]) + '_' + str(self.df.iloc[idx]["study_id"])]
                print(len(rows_new_labels_this_case),len(rows_new_labels_this_case)==0)
                if len(rows_new_labels_this_case)==0:
                    reflacx_present = 0
                    continue
                reflacx_present = 1
                print('oi1',len(rows_new_labels_this_case),reflacx_present)
            if len(rows_new_labels_this_case)==0:
                print(filepath)
            if dataset!='reflacx':
                assert len(rows_new_labels_this_case)==4
            else:
                assert len(rows_new_labels_this_case)==2

            this_location_labels = np.zeros([len(str_labels_location)])-2

            for index_label, label in enumerate(str_labels):
                if dataset=='reflacx':
                    if translation_mimic_to_new_labels[label] not in rows_new_labels_this_case.columns:
                        continue
                if dataset!='reflacx':
                    this_location_vector_index = -1
                    location = rows_new_labels_this_case[rows_new_labels_this_case['type_annotation']=='location'][translation_mimic_to_new_labels[label]].values[0]
                    if location!='-1' and location!='' and location!=-1 and location==location:
                        # print(location)
                        if dataset!='vqa': #TEMP
                            pass #TEMP
                            # this_location_vector_index = self.locations_indices_search[self.locations_indices_search['location_string']==location.replace(';', ', ')[:250]]['index_location'].values[0]
                        
                        for location_label_str in set(list_of_words + list(list_of_replacements.keys())):                    
                            regexp = re.compile(fr'\b{location_label_str}\b', flags=re.IGNORECASE)
                            if regexp.search(location):
                                if location_label_str in list_of_replacements.keys():
                                    words_found = list_of_replacements[location_label_str]
                                else:
                                    words_found = [location_label_str]
                                for replacement_word in words_found:
                                    if replacement_word in list_of_location_labels_per_abnormality[label]:
                                        this_location_labels[list_of_words.index(replacement_word)] = 1
                                    for uncertain_label in list_of_uncertain_labels_location[replacement_word]:
                                        if this_location_labels[list_of_words.index(uncertain_label)]!= 1:
                                            if uncertain_label in list_of_location_labels_per_abnormality[label]:
                                                this_location_labels[list_of_words.index(uncertain_label)] = -1
                                    for negative_label in list_of_negative_labels_location[replacement_word]:
                                        if this_location_labels[list_of_words.index(negative_label)]==-2:
                                            if negative_label in list_of_location_labels_per_abnormality[label]:
                                                this_location_labels[list_of_words.index(negative_label)] = 0
                if dataset!='reflacx':
                    severity = rows_new_labels_this_case[rows_new_labels_this_case['type_annotation']=='severity'][translation_mimic_to_new_labels[label]].values[0]
                    if severity== severity:               
                        severity = int(severity)
                    else:
                        severity = -1
                probability = rows_new_labels_this_case[rows_new_labels_this_case['type_annotation']=='probability'][translation_mimic_to_new_labels[label]].values[0]
                if probability== probability:   
                    probability = int(probability)
                else:
                    assert(dataset=='vqa')
                    probability=10
                new_label = int(rows_new_labels_this_case[rows_new_labels_this_case['type_annotation']=='labels'][translation_mimic_to_new_labels[label]].values[0])
                if dataset=='llm':
                    unchanged_uncertainty = ((probability==101) or (new_label==-3))* 1
                    unchanged_uncertainties[index_label] = unchanged_uncertainty
                    severities[index_label] = severity
                    probabilities[index_label] = probability
                    new_gt[index_label] = new_label
                    mimic_gt[index_label]= self.df.iloc[idx][label]
                    chexbert_gt[index_label]= self.df_chexbert[self.df_chexbert['image_1'] == (str(self.df.iloc[idx]['subject_id']) + '_' + str(self.df.iloc[idx]['study_id']))][label].values[0]
                    location_vector_index[index_label] = this_location_vector_index
                    location_labels[index_label,:] = this_location_labels
                elif dataset=='vqa':
                    vqa_severities[index_label] = severity
                    vqa_probabilities[index_label] = probability
                    vqa_new_gt[index_label] = new_label
                    vqa_location_vector_index[index_label] = this_location_vector_index
                    vqa_location_labels[index_label,:] = this_location_labels
                elif dataset=='reflacx':
                    reflacx_probabilities[index_label] = probability
                    reflacx_new_gt[index_label] = new_label
        
        return img, mimic_gt, chexbert_gt, new_gt, severities, location_labels, location_vector_index, probabilities, unchanged_uncertainties, vqa_new_gt, vqa_severities, vqa_location_labels, vqa_location_vector_index, vqa_probabilities, reflacx_new_gt, reflacx_probabilities, reflacx_present
    