# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# Ricardo Bigolin Lanfredi - 2023-14-12
# Description:
# Auxiliary file containing a pytorch dataset for the test set of the NIH dataset 
# using labels from the Pneumothorax and Pneumonia relabeling of the dataset

import imageio
import numpy as np
from torch.utils.data import Dataset
from list_labels import str_labels_mimic as str_labels
from global_paths import jpg_path, nih_dataset_location
import pandas as pd
from list_labels import str_labels_location, translation_mimic_to_new_labels
import re
from list_labels import str_labels_location as list_of_words
from list_labels import list_of_replacements_labels_location as list_of_replacements
from list_labels import list_of_uncertain_labels_location, list_of_negative_labels_location, list_of_location_labels_per_abnormality

from threading import Lock

s_print_lock = Lock()

class NIHCXRDataset(Dataset):
    def __init__(self, df, df_new_labels_llm, df_labels_pneumothorax, df_labels_pneumonia):
        self.df = df
        self.df_new_labels_llm = df_new_labels_llm
        self.df_labels_pneumothorax = df_labels_pneumothorax
        self.df_labels_pneumonia = df_labels_pneumonia

        # nih_correspondance = pd.read_csv('image_1_image_2_nih_conversion.csv')
        # nih_correspondance['image_1'] = nih_correspondance['image_1'].str.lstrip('./images/')
        # nih_correspondance['image_2'] = nih_correspondance['image_2'].str.rstrip('.jpg')
        self.df['image_1'] = self.df['Image Index']
        # self.df_new_labels_llm['image_2'] = self.df_new_labels_llm['subjectid_studyid'].str.lstrip('_')
        self.df_new_labels_llm['image_1'] = self.df_new_labels_llm['subjectid_studyid']
        cases = self.df_new_labels_llm[self.df_new_labels_llm['type_annotation']=='labels'][['image_1']]
        # self.df = pd.merge(self.df, nih_correspondance[['image_1','image_2']], on='image_1', how='inner')
        self.df = pd.merge(self.df, cases, on='image_1', how='inner')
        self.df['subjectid_studyid'] = self.df['image_1']
        
        # self.df_labels_pneumothorax['image_2'] = self.df_labels_pneumothorax['subjectid_studyid']
        self.df_labels_pneumothorax['image_1'] = self.df_labels_pneumothorax['subjectid_studyid']
        # self.df_labels_pneumonia['image_2'] = self.df_labels_pneumonia['subjectid_studyid']
        self.df_labels_pneumonia['image_1'] = self.df_labels_pneumonia['subjectid_studyid']

        # Cardiomegaly
        # Infiltration, Pneumonia
        # Mass, Nodule
        # Effusion
        # Atelectasis
        # Pneumothorax
        # Edema

        # self.df[str_labels] = (self.df[str_labels].fillna(-2))
        
        self.df_labels_pneumothorax = pd.merge(self.df_labels_pneumothorax, cases, on='image_1', how='inner')
        self.df_labels_pneumonia = pd.merge(self.df_labels_pneumonia, cases, on='image_1', how='inner')
        self.df_labels_pneumothorax['subjectid_studyid'] = self.df_labels_pneumothorax['image_1']
        self.df_labels_pneumonia['subjectid_studyid'] = self.df_labels_pneumonia['image_1']
        # self.locations_indices_search = pd.read_csv('clip_locations_embeddings.csv')
        self.dataset_size = len(self.df)
    
    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx):
        print(idx)
        if idx >= len(self):
            raise StopIteration
        
        filepath =  nih_dataset_location + str(self.df.iloc[idx]["image_1"])
        
        print(filepath)
        img = imageio.imread(filepath)
        if len(img.shape)==3:
            img = img[:,:,0]
        # mimic_gt = np.zeros([len(str_labels)])
        # new_gt = np.zeros([len(str_labels)])
        # probabilities = np.zeros([len(str_labels)])
        # location_labels = np.zeros([len(str_labels),len(str_labels_location)])-2
        # severities = np.zeros([len(str_labels)])
        # location_vector_index = np.zeros([len(str_labels)])-1
        # unchanged_uncertainties = np.zeros([len(str_labels)])

        # # llm
        # rows_new_labels_this_case = self.df_new_labels_llm[self.df_new_labels_llm['subjectid_studyid'] ==  str(self.df.iloc[idx]["subjectid_studyid"])]
        # if len(rows_new_labels_this_case)==0:
        #     print(filepath)
        # assert len(rows_new_labels_this_case)==4

        # this_location_labels = np.zeros([len(str_labels_location)])-2

        # for index_label, label in enumerate(str_labels):

        #     this_location_vector_index = -1
        #     location = rows_new_labels_this_case[rows_new_labels_this_case['type_annotation']=='location'][translation_mimic_to_new_labels[label]].values[0]
        #     if location!='-1' and location!='' and location!=-1 and location==location:
        #         this_location_vector_index = self.locations_indices_search[self.locations_indices_search['location_string']==location.replace(';', ', ')[:250]]['index_location'].values[0]
                
        #         for location_label_str in set(list_of_words + list(list_of_replacements.keys())):                    
        #             regexp = re.compile(fr'\b{location_label_str}\b', flags=re.IGNORECASE)
        #             if regexp.search(location):
        #                 if location_label_str in list_of_replacements.keys():
        #                     words_found = list_of_replacements[location_label_str]
        #                 else:
        #                     words_found = [location_label_str]
        #                 for replacement_word in words_found:
        #                     if replacement_word in list_of_location_labels_per_abnormality[label]:
        #                         this_location_labels[list_of_words.index(replacement_word)] = 1
        #                     for uncertain_label in list_of_uncertain_labels_location[replacement_word]:
        #                         if this_location_labels[list_of_words.index(uncertain_label)]!= 1:
        #                             if uncertain_label in list_of_location_labels_per_abnormality[label]:
        #                                 this_location_labels[list_of_words.index(uncertain_label)] = -1
        #                     for negative_label in list_of_negative_labels_location[replacement_word]:
        #                         if this_location_labels[list_of_words.index(negative_label)]==-2:
        #                             if negative_label in list_of_location_labels_per_abnormality[label]:
        #                                 this_location_labels[list_of_words.index(negative_label)] = 0
        #     severity = rows_new_labels_this_case[rows_new_labels_this_case['type_annotation']=='severity'][translation_mimic_to_new_labels[label]].values[0]
        #     if severity== severity:               
        #         severity = int(severity)
        #     else:
        #         severity = -1
        #     probability = rows_new_labels_this_case[rows_new_labels_this_case['type_annotation']=='probability'][translation_mimic_to_new_labels[label]].values[0]
        #     probability = int(probability)
        #     new_label = int(rows_new_labels_this_case[rows_new_labels_this_case['type_annotation']=='labels'][translation_mimic_to_new_labels[label]].values[0])
        #     unchanged_uncertainty = ((probability==101) or (new_label==-3))* 1
        #     unchanged_uncertainties[index_label] = unchanged_uncertainty
        #     severities[index_label] = severity
        #     probabilities[index_label] = probability
        #     new_gt[index_label] = new_label
        #     mimic_gt[index_label]= self.df.iloc[idx][label]
        #     location_vector_index[index_label] = this_location_vector_index
        #     location_labels[index_label,:] = this_location_labels
        
        #pneumothorax
        rows_new_labels_this_case = self.df_labels_pneumothorax[self.df_labels_pneumothorax['subjectid_studyid'] ==  str(self.df.iloc[idx]["subjectid_studyid"])]
        if len(rows_new_labels_this_case)==0:
            pneumothorax_label_present = 0
            pneumothorax_labels = 0
        else:
            pneumothorax_label_present = 1
            for index_label, label in enumerate(str_labels):
                pneumothorax_labels = int(rows_new_labels_this_case['pneumothorax'].values[0])
        
        #pneumonia
        rows_new_labels_this_case = self.df_labels_pneumonia[self.df_labels_pneumonia['subjectid_studyid'] ==  str(self.df.iloc[idx]["subjectid_studyid"])]
        if len(rows_new_labels_this_case)==0:
            pneumonia_label_present = 0
            pneumonia_labels = 0
        else:
            pneumonia_label_present = 1
            for index_label, label in enumerate(str_labels):
                pneumonia_labels = int(rows_new_labels_this_case['consolidation'].values[0])
        
        
        # return img, mimic_gt, new_gt, severities, location_labels, location_vector_index, probabilities, unchanged_uncertainties, pneumothorax_labels, pneumothorax_label_present, pneumonia_labels, pneumonia_label_present
        return img, pneumothorax_labels, pneumothorax_label_present, pneumonia_labels, pneumonia_label_present
