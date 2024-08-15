# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# Ricardo Bigolin Lanfredi - 2023-14-12
# This script takes clip_locations_embeddings.csv
# report_labels_mimic_0.0_llama2_all_tasks_generic_1_30_fixed.csv
# report_labels_mimic_0.0_llama2_all_tasks_1_30_fixed.csv
# vqa_dataset_converted.csv
# reflacx_dataset_converted.csv
# train_df.csv
# train_df_all.csv
# val_df.csv
# val_df_all.csv
# test_df.csv
# test_df_all.csv
# and MIMIC-CXR images from ./MIMIC-CXR/mimic-cxr-jpg-2.0.0.physionet.org/files/ as inputs
# and outputs results to ./dataset_statistics.csv
# Description:
# Script used to get the statistics of the CXR dataset annotations

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
allowed_locations = str_labels_location

from threading import Lock

s_print_lock = Lock()

str_labels_mimic = [
    'Enlarged Cardiomediastinum',
    'Cardiomegaly',
    'Lung Lesion',
    'Lung Opacity',
    'Edema',
    'Consolidation',
    'Atelectasis',
    'Pneumothorax',
    'Pleural Effusion',
    'Pleural Other',
    'Fracture',
    'Support Devices']

def pre_process_path(dicom_path):
    temp_path = jpg_path + '/files/' + dicom_path.split('files')[-1]
    temp_path = temp_path.replace('.dcm', '.jpg')
    return temp_path.strip()

class MIMICCXRDataset(Dataset):
    def __init__(self, df, chexbert_df, df_new_labels_llm, df_new_labels_vqa, df_labels_reflacx):
        self.df = df
        self.dataset_size = len(self.df)
        
        self.df[str_labels] = (self.df[str_labels].fillna(-2))
        # self.locations_indices_search = pd.read_csv('clip_locations_embeddings.csv')
        self.df_new_labels_llm = df_new_labels_llm
        self.df_new_labels_vqa = df_new_labels_vqa
        self.df_labels_reflacx = df_labels_reflacx
        self.df_chexbert = chexbert_df
        self.df['subjectid_studyid'] = self.df['subject_id'].astype(str) + '_' + self.df['study_id'].astype(str)
        self.df_new_labels_llm = pd.merge(self.df['subjectid_studyid'], self.df_new_labels_llm, on='subjectid_studyid', how='left').reset_index()
        self.df_new_labels_vqa = pd.merge(self.df['subjectid_studyid'], self.df_new_labels_vqa, on='subjectid_studyid', how='left').reset_index()
        # self.df_labels_reflacx = pd.merge(self.df['subjectid_studyid'], self.df_labels_reflacx, on='subjectid_studyid', how='left').reset_index()

    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx):
        if idx%100==0:
            print(idx)
        if idx >= len(self):
            raise StopIteration
        pid = 'p' + str(self.df.iloc[idx]["subject_id"])
        filepath = jpg_path + '/files/' + pid[:3] + '/' + pid + '/s' + str(self.df.iloc[idx]["study_id"]) + '/' + self.df.iloc[idx]["dicom_id"] + '.jpg'
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
                rows_new_labels_this_case = self.df_new_labels_llm.iloc[4*idx:4*(idx+1)]
                # rows_new_labels_this_case = self.df_new_labels_llm[self.df_new_labels_llm['subjectid_studyid'] ==  str(self.df.iloc[idx]["subject_id"]) + '_' + str(self.df.iloc[idx]["study_id"])]
            elif dataset=='vqa':
                rows_new_labels_this_case = self.df_new_labels_vqa.iloc[4*idx:4*(idx+1)]
                # rows_new_labels_this_case = self.df_new_labels_vqa[self.df_new_labels_vqa['subjectid_studyid'] ==  str(self.df.iloc[idx]["subject_id"]) + '_' + str(self.df.iloc[idx]["study_id"])]
            elif dataset=='reflacx':
                # rows_new_labels_this_case = self.df_labels_reflacx.iloc[2*idx:2*(idx+1)]
                rows_new_labels_this_case = self.df_labels_reflacx[self.df_labels_reflacx['subjectid_studyid'].astype(str) ==  str(self.df.iloc[idx]["subject_id"]) + '_' + str(self.df.iloc[idx]["study_id"])]
                # print(len(rows_new_labels_this_case),len(rows_new_labels_this_case)==0)
                # rows_new_labels_this_case = rows_new_labels_this_case[rows_new_labels_this_case['type_annotation']==rows_new_labels_this_case['type_annotation']]
                if len(rows_new_labels_this_case)==0:
                    reflacx_present = 0
                    continue
                reflacx_present = 1
                # print('oi1',len(rows_new_labels_this_case),reflacx_present)
            # print(rows_new_labels_this_case['subjectid_studyid'], str(self.df.iloc[idx]["subject_id"]) + '_' + str(self.df.iloc[idx]["study_id"]))
            assert((rows_new_labels_this_case['subjectid_studyid'] ==  str(self.df.iloc[idx]["subject_id"]) + '_' + str(self.df.iloc[idx]["study_id"])).all())
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
                    location_vector_index[index_label] = this_location_vector_index
                    location_labels[index_label,:] = this_location_labels
                    chexbert_gt[index_label]= self.df_chexbert[self.df_chexbert['image_1'] == (str(self.df.iloc[idx]['subject_id']) + '_' + str(self.df.iloc[idx]['study_id']))][label].values[0]

                elif dataset=='vqa':
                    vqa_severities[index_label] = severity
                    vqa_probabilities[index_label] = probability
                    vqa_new_gt[index_label] = new_label
                    vqa_location_vector_index[index_label] = this_location_vector_index
                    vqa_location_labels[index_label,:] = this_location_labels
                elif dataset=='reflacx':
                    reflacx_probabilities[index_label] = probability
                    reflacx_new_gt[index_label] = new_label
        return mimic_gt, chexbert_gt, new_gt, severities, location_labels, location_vector_index, probabilities, unchanged_uncertainties, vqa_new_gt, vqa_severities, vqa_location_labels, vqa_location_vector_index, vqa_probabilities, reflacx_new_gt, reflacx_probabilities, reflacx_present
        
def get_train_val_dfs(args):
    train_df = pd.read_csv(f'./train_df{"_all" if args.include_ap else ""}.csv')
    val_df = pd.read_csv(f'./val_df{"_all" if args.include_ap else ""}.csv')
    test_df = pd.read_csv(f'./test_df{"_all" if args.include_ap else ""}.csv')
    return train_df, val_df, test_df

def get_mimic_by_split(split,args):
    train_df, val_df, test_df = get_train_val_dfs(args)
    if args.do_generic:
        new_labels_df_llm = pd.read_csv('./mimic_llm_annotations_generic.csv')
    else:
        new_labels_df_llm = pd.read_csv('./new_dataset_annotations/mimic_llm_annotations.csv')
    new_labels_df_vqa = pd.read_csv('vqa_dataset_converted.csv')
    df_labels_reflacx = pd.read_csv('reflacx_dataset_converted.csv')
    chexbert_df = pd.read_csv('mimic_chexbert_labels.csv')
    return MIMICCXRDataset({'train': train_df, 'val': val_df, 'test': test_df}[split], chexbert_df, new_labels_df_llm, new_labels_df_vqa, df_labels_reflacx)
from types import SimpleNamespace
args = SimpleNamespace(do_generic = False, include_ap = True)
a = get_mimic_by_split('train',args)
table = []
for element in a:
    mimic_gt, chexbert_gt, new_gt, severities, location_labels, \
    location_vector_index, probabilities, unchanged_uncertainties, \
    vqa_new_gt, vqa_severities, vqa_location_labels, vqa_location_vector_index, \
    vqa_probabilities, reflacx_new_gt, reflacx_probabilities, reflacx_present = element
    
    for index_abnormality, abnormality in enumerate(str_labels):
        row = {}
        row['abnormality'] = abnormality
        row['mimic_gt'] = mimic_gt[index_abnormality]
        row['chexbert_gt'] = chexbert_gt[index_abnormality]
        row['new_gt'] = new_gt[index_abnormality]
        row['vqa_new_gt'] = vqa_new_gt[index_abnormality]
        row['reflacx_new_gt'] = reflacx_new_gt[index_abnormality]
        labels_dict = {'stable':-3,'not mentioned':-2,'uncertain':-1,'absent':0,'present':1}
        for category in labels_dict:
            row['mimic_gt_'+category] = (mimic_gt[index_abnormality]==labels_dict[category])*1
            row['new_gt_'+category] = (new_gt[index_abnormality]==labels_dict[category])*1
            row['vqa_new_gt_'+category] = (vqa_new_gt[index_abnormality]==labels_dict[category])*1
            row['reflacx_new_gt_'+category] = (reflacx_new_gt[index_abnormality]==labels_dict[category])*1
        row['severities'] = severities[index_abnormality]
        row['vqa_severities'] = vqa_severities[index_abnormality]
        row['severities_present'] = (severities[index_abnormality]>0)*1
        row['vqa_severities_present'] = (vqa_severities[index_abnormality]>0)*1
        severities_dict = {'mild':1,'moderate':2,'severe':3}
        for category in severities_dict:
            row['severities_'+category] = (severities[index_abnormality]==severities_dict[category])*1
            row['vqa_severities_'+category] = (vqa_severities[index_abnormality]==severities_dict[category])*1
        row['probabilities'] = probabilities[index_abnormality]
        row['vqa_probabilities'] = vqa_probabilities[index_abnormality]
        row['reflacx_probabilities'] = reflacx_probabilities[index_abnormality]
        location_labels_abnormality = location_labels[index_abnormality]
        location_labels_abnormality = np.array([location_labels_abnormality[allowed_locations.index(allowed_location)] for allowed_location in allowed_locations if allowed_location in list_of_location_labels_per_abnormality[abnormality]])
        assert(len(location_labels_abnormality)<32)
        row['location_labels_positive'] = (location_labels_abnormality==1).sum()
        row['location_labels_negative'] = (location_labels_abnormality==0).sum()
        row['location_labels'] = len(location_labels_abnormality)
        vqa_location_labels_abnormality = vqa_location_labels[index_abnormality]
        vqa_location_labels_abnormality = np.array([vqa_location_labels_abnormality[allowed_locations.index(allowed_location)] for allowed_location in allowed_locations if allowed_location in list_of_location_labels_per_abnormality[abnormality]])
        assert(len(vqa_location_labels_abnormality)<32)
        row['vqa_location_labels_positive'] = (vqa_location_labels_abnormality==1).sum()
        row['vqa_location_labels_negative'] = (vqa_location_labels_abnormality==0).sum()
        row['vqa_location_labels'] = len(vqa_location_labels_abnormality)
        row['unchanged_uncertainties'] = unchanged_uncertainties[index_abnormality]
        row['reflacx_present'] = reflacx_present
        table.append(row)
pd.DataFrame(table).to_csv('./dataset_statistics.csv')

        

