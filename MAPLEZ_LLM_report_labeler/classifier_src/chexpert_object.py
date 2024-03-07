# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# Ricardo Bigolin Lanfredi - 2023-14-12
# Description:
# Auxiliary file containing a pytorch dataset for the CheXpert test dataset

import imageio
import numpy as np
from torch.utils.data import Dataset
from list_labels import str_labels_mimic as str_labels
from global_paths import jpg_path, chexpert_dataset_location
import pandas as pd
from list_labels import str_labels_location, translation_mimic_to_new_labels
import re
from list_labels import str_labels_location as list_of_words
from list_labels import list_of_replacements_labels_location as list_of_replacements
from list_labels import list_of_uncertain_labels_location, list_of_negative_labels_location, list_of_location_labels_per_abnormality

from threading import Lock

s_print_lock = Lock()

# atelectasis, cardiomegaly,
# consolidation, edema, and pleural effusion

class ChexpertCXRDataset(Dataset):
    def __init__(self, df):
        self.df = df
        
        if 'Frontal/Lateral' in self.df.columns:
            self.df = self.df[self.df['Frontal/Lateral']=='Frontal']

        self.df[str_labels] = (self.df[str_labels].fillna(-2))
        self.dataset_size = len(self.df)
    
    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx):
        print(idx)
        if idx >= len(self):
            raise StopIteration
        
        filepath =  chexpert_dataset_location+ str(self.df.iloc[idx]["Path"]).replace('-v1.0','').replace('/valid/','/val/')
        if not '_frontal.jpg' in filepath:
            filepath = filepath + '/view1_frontal.jpg'
        
        img = imageio.imread(filepath)
        mimic_gt = np.zeros([len(str_labels)])

        for index_label, label in enumerate(str_labels):
            mimic_gt[index_label]= self.df.iloc[idx][label]
        
        return img, mimic_gt